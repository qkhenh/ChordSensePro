"""ProcessAudioService — DAG 2 service: process WAV from MongoDB → save to PostgreSQL.

Responsibility: Transform + Load (T+L phases).
Reads wav_path + annotation_path from RawAudioJob, runs training pipeline
(Separate → Beat → Segment → Annotation → Augment → Feature), saves to PostgreSQL,
then cleans up temp files from disk.
"""
from __future__ import annotations
import shutil
from pathlib import Path

from src.data_loader.domain.models.audio_file import AudioFile, AudioSource
from src.data_processing.domain.models.processed_audio import ProcessedAudio
from src.data_processing.application.pipeline_factory import build_training_pipeline
from src.data_ingest.domain.models.raw_audio_job import RawAudioJob, RawJobStatus
from src.data_ingest.domain.models.ingest_record import IngestRecord, IngestStatus
from src.data_ingest.infrastructure.mongo.raw_audio_job_repository import RawAudioJobRepository
from src.song_analysis.domain.models.song_analysis import SongAnalysis, AnalysisStatus
from src.song_analysis.infrastructure.repositories.song_analysis_repository import SongAnalysisRepository
from src.shared.infrastructure.mongo.client import get_mongo_db


class ProcessAudioService:
    """Processes one RawAudioJob through the full T→L pipeline.

    Called by DAG 2 (dag_process_audio).
    Reads wav_path + annotation_path from MongoDB job, skips download entirely.
    Cleans up temp files after saving to PostgreSQL.
    """

    async def run(self, job: RawAudioJob) -> IngestRecord:
        """Transform WAV → stems → features → chord timeline, save to PostgreSQL.

        Args:
            job: RawAudioJob with wav_path (and optionally annotation_path) on disk.

        Returns:
            IngestRecord with final status.
        """
        mongo_repo = RawAudioJobRepository(get_mongo_db())
        sa_repo    = SongAnalysisRepository()
        record     = IngestRecord(analysis_id=job.id, status=IngestStatus.PROCESSING)

        # Mark as PROCESSING to prevent double-pickup
        await mongo_repo.update_status(job.id, RawJobStatus.PROCESSING)

        # ── Check cache — skip if already analyzed ───────────────────────────
        cached = sa_repo.get_by_url(job.source_url)
        if cached.is_ok:
            await mongo_repo.update_status(job.id, RawJobStatus.DONE)
            record.status = IngestStatus.DONE
            return record

        # ── Validate WAV exists ───────────────────────────────────────────────
        wav_path = Path(job.wav_path)
        if not wav_path.exists():
            error = f"WAV not found at {wav_path} — file may have been deleted"
            await mongo_repo.update_status(job.id, RawJobStatus.FAILED, error)
            record.fail(error)
            return record

        # ── T: Transform — training pipeline ──────────────────────────────────
        audio_file = AudioFile(
            source_url=job.source_url,
            analysis_id=job.id,
            source_type=AudioSource(job.source_type),
            dest_dir=wav_path.parent,
        )
        data = ProcessedAudio(audio_file=audio_file, wav_path=wav_path)

        # Set annotation_path so AnnotationHandler can parse chord labels
        if job.annotation_path:
            data.annotation_path = Path(job.annotation_path)

        pipeline  = build_training_pipeline()
        processed = pipeline.handle(data)

        if processed.is_failed:
            await mongo_repo.update_status(job.id, RawJobStatus.FAILED, processed.error)
            record.fail(processed.error or "Pipeline failed")
            self._cleanup(wav_path)
            return record

        # ── L: Load to PostgreSQL ─────────────────────────────────────────────
        record.status = IngestStatus.SAVING
        analysis = SongAnalysis(
            id=job.id,
            user_id="",
            source_url=job.source_url,
            song_title="",
            status=AnalysisStatus.PROCESSING,
            tempo_bpm=processed.tempo_bpm,
        )
        analysis.mark_done(
            key=getattr(processed, "detected_key", "") or "",
            bpm=processed.tempo_bpm or 0.0,
            timeline=getattr(processed, "chord_timeline", []) or [],
            sheet=getattr(processed, "chord_sheet", {}) or {},
            plan=[],
        )
        sa_repo.save_domain(analysis)

        await mongo_repo.update_status(job.id, RawJobStatus.DONE)
        record.status = IngestStatus.DONE

        # ── Cleanup: remove temp files to free disk space ────────────────────
        self._cleanup(wav_path)
        return record

    @staticmethod
    def _cleanup(wav_path: Path) -> None:
        """Delete the job temp directory (WAV + stems + segments) after processing.

        Non-fatal — failure is logged but does not affect pipeline result.
        """
        try:
            shutil.rmtree(wav_path.parent, ignore_errors=True)
        except Exception:
            pass  # best-effort cleanup
