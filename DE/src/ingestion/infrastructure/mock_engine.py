"""Mock ``AnalysisEngine``: answers from the golden fixtures and simulates the job lifecycle.

Scenario is chosen from the stem of ``audio_path`` (the Normalizer names the WAV after the
sanitised upload name, so a developer controls it with the file name):

* ``demo-<fixture>``  -> exactly that fixture (used by the demo catalog)
* contains ``fail-ai``      -> ``AnalysisError(AI_FAILURE)``
* contains ``fail-timeout`` -> never finishes (exercises the 90 s timeout)
* contains ``slow``         -> every step takes ``slow_step_ms`` (default 1 s): a job long enough
  to reload the page in the middle of
* anything else       -> fixture ``int(content_hash, 16) % 7``
"""

import asyncio
from pathlib import Path

from src.analysis_contract.models import (
    AnalysisRequest,
    AnalysisResult,
    ErrorCode,
    JobProgress,
    JobStage,
    JobState,
)
from src.analysis_contract.ports import AnalysisError, ProgressCallback

FIXTURES_DIR = Path(__file__).parents[4] / "contracts" / "fixtures" / "results"
_STEPS = ((JobStage.SEPARATING, 0.45), (JobStage.BEAT_TRACKING, 0.7), (JobStage.RECOGNIZING, 0.9))


class MockAnalysisEngine:
    def __init__(
        self, fixtures_dir: Path = FIXTURES_DIR, stage_delay_ms: int = 0, slow_step_ms: int = 1000
    ) -> None:
        self._delay = stage_delay_ms / 1000
        self._slow_delay = slow_step_ms / 1000
        self._results: dict[str, AnalysisResult] = {
            path.name.removesuffix(".result.json"): AnalysisResult.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            for path in sorted(fixtures_dir.glob("*.result.json"))
        }
        self._names = list(self._results)

    def _pick(self, request: AnalysisRequest) -> AnalysisResult:
        stem = Path(request.audio_path).stem
        if stem.startswith("demo-") and stem.removeprefix("demo-") in self._results:
            return self._results[stem.removeprefix("demo-")]
        return self._results[self._names[int(request.content_hash, 16) % len(self._names)]]

    async def analyze(
        self, request: AnalysisRequest, on_progress: ProgressCallback
    ) -> AnalysisResult:
        stem = Path(request.audio_path).stem
        if "fail-timeout" in stem:
            await asyncio.sleep(3600)
        delay = max(self._delay, self._slow_delay) if "slow" in stem else self._delay
        for stage, progress in _STEPS:
            if delay:
                await asyncio.sleep(delay)
            await on_progress(
                JobProgress(
                    job_id=request.job_id,
                    state=JobState.PROCESSING,
                    stage=stage,
                    progress=progress,
                )
            )
            if "fail-ai" in stem and stage == JobStage.BEAT_TRACKING:
                raise AnalysisError(
                    ErrorCode.AI_FAILURE, "The analysis model failed on this audio", retryable=True
                )
        return self._pick(request).model_copy(
            update={"job_id": request.job_id, "content_hash": request.content_hash}
        )
