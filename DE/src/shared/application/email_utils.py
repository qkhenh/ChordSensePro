"""Email notification utility — sends SMTP emails via EmailSettings.

Used by DAG wrappers to notify on success/failure.
If SMTP credentials are not configured in .env, emails are silently skipped.
"""
from __future__ import annotations
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

log = logging.getLogger(__name__)


def send_email(subject: str, body: str) -> None:
    """Send an email using SMTP credentials from .env/config.

    Silent no-op if email is not configured (SMTP_USER or SMTP_PASSWORD empty).
    """
    try:
        from src.shared.infrastructure.settings.config import get_settings
        cfg = get_settings().email  # type: ignore[attr-defined]
    except AttributeError:
        log.info("[email] SMTP not configured (no email settings) — skipping notification")
        return

    if not cfg.enabled:
        log.info("[email] SMTP not configured — skipping notification")
        return

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = cfg.user
        msg["To"]      = cfg.to

        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(cfg.host, cfg.port) as server:
            server.ehlo()
            server.starttls()
            server.login(cfg.user, cfg.password)
            server.sendmail(cfg.user, cfg.to.split(","), msg.as_string())

        log.info(f"[email] Sent: {subject}")
    except Exception as e:
        log.warning(f"[email] Failed to send email: {e}")


def notify_dag_failure(dag_id: str, task_id: str, error: str) -> None:
    """Standard failure notification — called by on_failure_callback."""
    send_email(
        subject=f"🔴 [ChordSense] DAG Failed: {dag_id}/{task_id}",
        body=f"DAG: {dag_id}\nTask: {task_id}\n\nError:\n{error}",
    )


def notify_batch_summary(dag_id: str, done: int, failed: int, skipped: int) -> None:
    """Batch completion summary email."""
    status = "✅" if failed == 0 else "⚠️"
    send_email(
        subject=f"{status} [ChordSense] {dag_id} — {done} done, {failed} failed",
        body=(
            f"DAG: {dag_id}\n\n"
            f"Results:\n"
            f"  ✓ Done:    {done}\n"
            f"  ✗ Failed:  {failed}\n"
            f"  ~ Skipped: {skipped}\n"
            f"  Total:     {done + failed + skipped}\n"
        ),
    )
