from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MarginalLinearChainCRF(nn.Module):
    """Linear-Chain CRF over `num_tags` states supporting exact marginalization over `-100` unlabelled gaps."""

    def __init__(self, num_tags: int = 145, stay_bonus: float = 2.4):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.zeros(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.zeros(num_tags))
        self.end_transitions = nn.Parameter(torch.zeros(num_tags))
        self._init_musical_prior(stay_bonus)

    def _init_musical_prior(self, stay_bonus: float) -> None:
        with torch.no_grad():
            self.transitions.zero_()
            idx = torch.arange(self.num_tags)
            self.transitions[idx, idx] = stay_bonus
            # Mild harmonic affinity between Skeleton states sharing the same Root (e.g. V7sus4 -> V7, Am -> Am7)
            if self.num_tags == 145:
                for r in range(12):
                    lo = 1 + r * 12
                    hi = lo + 12
                    block = self.transitions[lo:hi, lo:hi]
                    block += 0.35
                    block[torch.arange(12), torch.arange(12)] = stay_bonus

    def forward(self, emissions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute exact per-observed-frame negative log marginal likelihood in float32.

        Args:
            emissions: `(B, T, K)` unnormalized state scores.
            targets: `(B, T)` int64 state IDs in `0..K-1`, or `-100` for unobserved/ignored steps.
        """
        em = emissions.float()
        bsz, seq_len, num_tags = em.shape
        obs_mask = (targets >= 0) & (targets < num_tags)
        n_obs = obs_mask.sum(dim=1).float()  # (B,)
        if not obs_mask.any():
            return em.sum() * 0.0

        trans = self.transitions.float()  # (K, K): from_state -> to_state
        start_t = self.start_transitions.float()
        end_t = self.end_transitions.float()

        # Constrained emissions: 0 penalty on ground-truth state when observed, 0 on ALL states when -100
        neg_inf = -1e9
        constraint = torch.zeros_like(em)
        if obs_mask.any():
            safe_tgt = targets.clamp(0, num_tags - 1)
            one_hot = torch.zeros_like(em).scatter_(-1, safe_tgt.unsqueeze(-1), 1.0)
            obs_pen = torch.where(one_hot > 0.5, 0.0, neg_inf)
            constraint = torch.where(obs_mask.unsqueeze(-1), obs_pen, 0.0)

        em_clamped = em + constraint

        alpha = start_t.unsqueeze(0) + em[:, 0, :]          # (B, K) unconstrained partition
        beta = start_t.unsqueeze(0) + em_clamped[:, 0, :]   # (B, K) target-constrained partition
        trans_b = trans.unsqueeze(0)                        # (1, K_prev, K_next)

        for t in range(1, seq_len):
            alpha = em[:, t, :] + torch.logsumexp(alpha.unsqueeze(2) + trans_b, dim=1)
            beta = em_clamped[:, t, :] + torch.logsumexp(beta.unsqueeze(2) + trans_b, dim=1)

        log_z = torch.logsumexp(alpha + end_t.unsqueeze(0), dim=1)
        log_gold = torch.logsumexp(beta + end_t.unsqueeze(0), dim=1)

        valid_seq = n_obs > 0
        nll = (log_z - log_gold).clamp(min=0.0) / n_obs.clamp(min=1.0)
        return nll[valid_seq].mean()

    @torch.no_grad()
    def decode(self, emissions: torch.Tensor) -> torch.Tensor:
        """Vectorized batch Viterbi decoding returning `(B, T)` int64 state IDs."""
        em = emissions.float()
        bsz, seq_len, num_tags = em.shape
        trans = self.transitions.float().unsqueeze(0)  # (1, K_prev, K_next)
        dp = self.start_transitions.float().unsqueeze(0) + em[:, 0, :]
        backptrs: list[torch.Tensor] = []

        for t in range(1, seq_len):
            scores = dp.unsqueeze(2) + trans  # (B, K_prev, K_next)
            best_scores, best_idx = scores.max(dim=1)
            backptrs.append(best_idx)
            dp = best_scores + em[:, t, :]

        dp = dp + self.end_transitions.float().unsqueeze(0)
        best_last = dp.argmax(dim=-1)  # (B,)

        paths = [best_last]
        cur = best_last
        for bp in reversed(backptrs):
            cur = bp.gather(1, cur.unsqueeze(1)).squeeze(1)
            paths.append(cur)
        paths.reverse()
        return torch.stack(paths, dim=1)


def viterbi_crf_numpy(
    emissions: np.ndarray,
    transitions: np.ndarray,
    start_transitions: np.ndarray | None = None,
    end_transitions: np.ndarray | None = None,
    decay_lock_scale: np.ndarray | None = None,
) -> np.ndarray:
    """Fast NumPy Viterbi decoder for `(T, K)` CRF emissions with optional Onset-Decay Locking."""
    seq_len, num_tags = emissions.shape
    if seq_len == 0:
        return np.empty(0, dtype=np.int64)

    em = emissions.astype(np.float64)
    trans = transitions.astype(np.float64)
    off_diag = ~np.eye(num_tags, dtype=bool)

    dp = em[0].copy()
    if start_transitions is not None:
        dp += start_transitions.astype(np.float64)

    backptr = np.zeros((seq_len, num_tags), dtype=np.int32)
    for t in range(1, seq_len):
        step_trans = trans
        if decay_lock_scale is not None and decay_lock_scale[t] > 1.0:
            # During decaying chord tails (no new onset), penalize switching to a different chord state
            # except transitioning cleanly into state 0 (`N` = silence).
            extra_pen = float(decay_lock_scale[t] - 1.0) * 2.5
            step_trans = trans.copy()
            mask = off_diag.copy()
            mask[:, 0] = False
            step_trans[mask] -= extra_pen

        scores = dp[:, None] + step_trans  # (K_prev, K_next)
        best_prev = scores.argmax(axis=0)
        backptr[t] = best_prev
        dp = scores[best_prev, np.arange(num_tags)] + em[t]

    if end_transitions is not None:
        dp += end_transitions.astype(np.float64)

    path = np.empty(seq_len, dtype=np.int64)
    path[-1] = int(dp.argmax())
    for t in range(seq_len - 1, 0, -1):
        path[t - 1] = backptr[t, path[t]]
    return path


class DynamicBoundaryCRF(nn.Module):
    """Boundary-Conditioned Linear-Chain CRF over `num_tags` states (`169` for Model v6).

    At each step `t` with predicted chord-boundary probability `b_t in [0, 1]`:
      - Self-transition (stay) penalty is softened: `A_stay(t) = A_stay * (1 - stay_scale * b_t)`
      - Off-diagonal chord transitions receive a boundary boost: `A_off(t) = A_off + trans_boost * b_t`

    This allows rapid passing chords (`240–750ms`) with genuine acoustic onsets to switch cleanly
    while keeping sustained bars rock-solid (`b_t ~ 0`) without over-smoothing.
    Also keeps a frozen `prior_transitions` buffer for L2 regularization against overfitting.
    """

    def __init__(
        self,
        num_tags: int = 169,
        stay_bonus: float = 2.4,
        prior_matrix: np.ndarray | None = None,
        boundary_stay_scale: float = 0.85,
        boundary_trans_boost: float = 1.25,
    ):
        super().__init__()
        self.num_tags = num_tags
        if prior_matrix is not None:
            prior_t = torch.tensor(prior_matrix, dtype=torch.float32)
        else:
            prior_t = torch.zeros(num_tags, num_tags, dtype=torch.float32)
            idx = torch.arange(num_tags)
            prior_t[idx, idx] = stay_bonus

        self.register_buffer("prior_transitions", prior_t.clone())
        self.register_buffer("diag_mask", torch.eye(num_tags, dtype=torch.float32), persistent=False)
        self.transitions = nn.Parameter(prior_t.clone())
        self.start_transitions = nn.Parameter(torch.zeros(num_tags))
        self.end_transitions = nn.Parameter(torch.zeros(num_tags))
        self.boundary_stay_scale = nn.Parameter(torch.tensor(float(boundary_stay_scale)))
        self.boundary_trans_boost = nn.Parameter(torch.tensor(float(boundary_trans_boost)))

    def prior_reg_loss(self) -> torch.Tensor:
        """L2 anchor pulling learned CRF transitions gently toward the music-theory functional prior."""
        return torch.mean((self.transitions.float() - self.prior_transitions.float()) ** 2)

    def _effective_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        stay_s = self.boundary_stay_scale.float().clamp(0.20, 0.98)
        boost = self.boundary_trans_boost.float().clamp(0.0, 3.0)
        return stay_s, boost

    def forward(
        self,
        emissions: torch.Tensor,
        targets: torch.Tensor,
        boundary_prob: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute exact per-observed-frame NLL with dynamic boundary-modulated transitions."""
        em = emissions.float()
        bsz, seq_len, num_tags = em.shape
        obs_mask = (targets >= 0) & (targets < num_tags)
        n_obs = obs_mask.sum(dim=1).float()
        if not obs_mask.any():
            return em.sum() * 0.0

        trans = self.transitions.float()  # (K, K)
        start_t = self.start_transitions.float()
        end_t = self.end_transitions.float()
        diag_m = self.diag_mask.float()
        off_m = 1.0 - diag_m
        stay_s, boost = self._effective_params()

        neg_inf = -1e9
        safe_tgt = targets.clamp(0, num_tags - 1)
        one_hot = torch.zeros_like(em).scatter_(-1, safe_tgt.unsqueeze(-1), 1.0)
        obs_pen = torch.where(one_hot > 0.5, 0.0, neg_inf)
        em_clamped = em + torch.where(obs_mask.unsqueeze(-1), obs_pen, 0.0)

        alpha = start_t.unsqueeze(0) + em[:, 0, :]
        beta = start_t.unsqueeze(0) + em_clamped[:, 0, :]

        diag_vals = trans * diag_m  # (K, K) diagonal only
        if boundary_prob is not None:
            b_prob = boundary_prob.float().clamp(0.0, 1.0)  # (B, T)
        else:
            b_prob = None

        for t in range(1, seq_len):
            if b_prob is not None:
                bt = b_prob[:, t].view(bsz, 1, 1)  # (B, 1, 1)
                step_trans = trans.unsqueeze(0) - (stay_s * bt) * diag_vals.unsqueeze(0) + (boost * bt) * off_m.unsqueeze(0)
            else:
                step_trans = trans.unsqueeze(0)
            alpha = em[:, t, :] + torch.logsumexp(alpha.unsqueeze(2) + step_trans, dim=1)
            beta = em_clamped[:, t, :] + torch.logsumexp(beta.unsqueeze(2) + step_trans, dim=1)

        log_z = torch.logsumexp(alpha + end_t.unsqueeze(0), dim=1)
        log_gold = torch.logsumexp(beta + end_t.unsqueeze(0), dim=1)
        valid_seq = n_obs > 0
        nll = (log_z - log_gold).clamp(min=0.0) / n_obs.clamp(min=1.0)
        return nll[valid_seq].mean()

    @torch.no_grad()
    def decode(
        self,
        emissions: torch.Tensor,
        boundary_prob: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Vectorized batch Viterbi decoding returning `(B, T)` int64 state IDs."""
        em = emissions.float()
        bsz, seq_len, num_tags = em.shape
        trans = self.transitions.float()
        diag_m = self.diag_mask.float()
        off_m = 1.0 - diag_m
        diag_vals = trans * diag_m
        stay_s, boost = self._effective_params()
        b_prob = boundary_prob.float().clamp(0.0, 1.0) if boundary_prob is not None else None

        dp = self.start_transitions.float().unsqueeze(0) + em[:, 0, :]
        backptrs: list[torch.Tensor] = []

        for t in range(1, seq_len):
            if b_prob is not None:
                bt = b_prob[:, t].view(bsz, 1, 1)
                step_trans = trans.unsqueeze(0) - (stay_s * bt) * diag_vals.unsqueeze(0) + (boost * bt) * off_m.unsqueeze(0)
            else:
                step_trans = trans.unsqueeze(0)
            scores = dp.unsqueeze(2) + step_trans
            best_scores, best_idx = scores.max(dim=1)
            backptrs.append(best_idx)
            dp = best_scores + em[:, t, :]

        dp = dp + self.end_transitions.float().unsqueeze(0)
        best_last = dp.argmax(dim=-1)

        paths = [best_last]
        cur = best_last
        for bp in reversed(backptrs):
            cur = bp.gather(1, cur.unsqueeze(1)).squeeze(1)
            paths.append(cur)
        paths.reverse()
        return torch.stack(paths, dim=1)


def viterbi_dynamic_crf_numpy(
    emissions: np.ndarray,
    transitions: np.ndarray,
    start_transitions: np.ndarray | None = None,
    end_transitions: np.ndarray | None = None,
    boundary_prob: np.ndarray | None = None,
    boundary_stay_scale: float = 0.85,
    boundary_trans_boost: float = 1.25,
    decay_lock_scale: np.ndarray | None = None,
) -> np.ndarray:
    """Fast NumPy Viterbi decoder matching `DynamicBoundaryCRF.decode` with optional decay-tail lock."""
    seq_len, num_tags = emissions.shape
    if seq_len == 0:
        return np.empty(0, dtype=np.int64)

    em = emissions.astype(np.float64)
    trans = transitions.astype(np.float64)
    diag_idx = np.arange(num_tags)
    diag_vals = trans[diag_idx, diag_idx].copy()
    off_diag = ~np.eye(num_tags, dtype=bool)
    stay_s = float(np.clip(boundary_stay_scale, 0.20, 0.98))
    boost = float(np.clip(boundary_trans_boost, 0.0, 3.0))

    dp = em[0].copy()
    if start_transitions is not None:
        dp += start_transitions.astype(np.float64)

    backptr = np.zeros((seq_len, num_tags), dtype=np.int32)
    for t in range(1, seq_len):
        step_trans = trans
        need_copy = False
        bt = float(np.clip(boundary_prob[t], 0.0, 1.0)) if boundary_prob is not None else 0.0
        has_decay = decay_lock_scale is not None and float(decay_lock_scale[t]) > 1.0 and bt < 0.35

        if bt > 1e-4 or has_decay:
            step_trans = trans.copy()
            need_copy = True

        if need_copy and bt > 1e-4:
            step_trans[off_diag] += boost * bt
            step_trans[diag_idx, diag_idx] -= stay_s * bt * diag_vals

        if need_copy and has_decay:
            extra_pen = float(decay_lock_scale[t] - 1.0) * 2.5
            mask = off_diag.copy()
            mask[:, 0] = False
            step_trans[mask] -= extra_pen

        scores = dp[:, None] + step_trans
        best_prev = scores.argmax(axis=0)
        backptr[t] = best_prev
        dp = scores[best_prev, diag_idx] + em[t]

    if end_transitions is not None:
        dp += end_transitions.astype(np.float64)

    path = np.empty(seq_len, dtype=np.int64)
    path[-1] = int(dp.argmax())
    for t in range(seq_len - 1, 0, -1):
        path[t - 1] = backptr[t, path[t]]
    return path
