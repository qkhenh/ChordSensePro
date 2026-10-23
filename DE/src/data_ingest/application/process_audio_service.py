"""ProcessAudioService — DAG 2 service: process WAV from MongoDB → save training samples.

Responsibility: Transform + Load (T+L phases).
Reads wav_path + annotation_path from RawAudioJob, runs the training pipeline
(AlignmentCheck → Separate → Beat → Segment → Annotation → Label → SampleWriter),
stores the clean labeled 2s segments as an int16 .npy on the training volume and their
metadata/labels in PostgreSQL ``training_samples``, then cleans up temp files.

The previous version wrote a song-level ``SongAnalysis`` row; that belongs to the system
(inference) side and is no longer produced by the training ETL.
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
from src.shared.infrastructure.mongo.client import get_mongo_db
from src.training_data.domain.models.training_sample import TrainingSample
from src.training_data.infrastructure.repositories.training_sample_repository import TrainingSampleRepository


class ProcessAudioService:
    """Processes one RawAudioJob through the full T→L pipeline.

    Called by DAG 2 (dag_process_audio).
    Reads wav_path + annotation_path from MongoDB job, skips download entirely.
    Cleans up temp files after saving to PostgreSQL.
    """

    async def run(self, job: RawAudioJob) -> IngestRecord:
        """Transform WAV → labeled clean segments, save to disk + PostgreSQL.

        Args:
            job: RawAudioJob with wav_path (and optionally annotation_path) on disk.

        Returns:
            IngestRecord with final status.
        """
        mongo_repo = RawAudioJobRepository(get_mongo_db())
        ts_repo    = TrainingSampleRepository()
        ts_repo.ensure_table()
        record     = IngestRecord(analysis_id=job.id, status=IngestStatus.PROCESSING)
        wav_path   = Path(job.wav_path)

        # Mark as PROCESSING to prevent double-pickup
        await mongo_repo.update_status(job.id, RawJobStatus.PROCESSING)

        # ── Check cache — skip if this source already produced samples ───────
        if ts_repo.source_exists(job.source_url):
            await mongo_repo.update_status(job.id, RawJobStatus.DONE)
            record.status = IngestStatus.DONE
            self._cleanup(wav_path)
            return record

        # ── Validate WAV exists ───────────────────────────────────────────────
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

        processed = build_training_pipeline().handle(data)

        if processed.is_failed:
            await mongo_repo.update_status(job.id, RawJobStatus.FAILED, processed.error)
            record.fail(processed.error or "Pipeline failed")
            self._cleanup(wav_path)
            return record

        # ── L: Load to PostgreSQL ─────────────────────────────────────────────
        record.status = IngestStatus.SAVING
        samples = [TrainingSample(**row) for row in processed.sample_rows]
        try:
            ts_repo.replace_song(job.id, samples)
        except Exception as e:
            error = f"Saving training_samples failed: {e}"
            await mongo_repo.update_status(job.id, RawJobStatus.FAILED, error)
            record.fail(error)
            self._cleanup(wav_path)
            return record

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
