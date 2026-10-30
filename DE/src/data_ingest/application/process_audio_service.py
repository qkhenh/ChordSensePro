from __future__ import annotations
import shutil
from pathlib import Path

from src.data_loader.domain.models.audio_file import AudioFile, AudioSource
from src.data_processing.domain.models.processed_audio import ProcessedAudio
from src.data_processing.application.pipeline_factory import build_training_pipeline, build_training_pipeline_v2
from src.data_ingest.domain.models.raw_audio_job import RawAudioJob, RawJobStatus
from src.data_ingest.domain.models.ingest_record import IngestRecord, IngestStatus
from src.data_ingest.infrastructure.mongo.raw_audio_job_repository import RawAudioJobRepository
from src.shared.infrastructure.mongo.client import get_mongo_db
from src.training_data.domain.models.training_sample import TrainingSample
from src.training_data.domain.models.training_section import TrainingSection, TrainingChord
from src.training_data.infrastructure.repositories.training_sample_repository import TrainingSampleRepository
from src.training_data.infrastructure.repositories.training_section_repository import TrainingSectionRepository


class ProcessAudioService:

    def __init__(self, pipeline_version: int = 2) -> None:
        self.pipeline_version = pipeline_version

    async def run(self, job: RawAudioJob) -> IngestRecord:
        mongo_repo = RawAudioJobRepository(get_mongo_db())
        record     = IngestRecord(analysis_id=job.id, status=IngestStatus.PROCESSING)
        wav_path   = Path(job.wav_path)

        await mongo_repo.update_status(job.id, RawJobStatus.PROCESSING)

        if self.pipeline_version == 2:
            ts_repo = TrainingSectionRepository()
            ts_repo.ensure_tables()
            if ts_repo.source_exists(job.source_url):
                await mongo_repo.update_status(job.id, RawJobStatus.DONE)
                record.status = IngestStatus.DONE
                self._cleanup(wav_path)
                return record
        else:
            ts_repo_v1 = TrainingSampleRepository()
            ts_repo_v1.ensure_table()
            if ts_repo_v1.source_exists(job.source_url):
                await mongo_repo.update_status(job.id, RawJobStatus.DONE)
                record.status = IngestStatus.DONE
                self._cleanup(wav_path)
                return record

        if not wav_path.exists():
            error = f"WAV not found at {wav_path} — file may have been deleted"
            await mongo_repo.update_status(job.id, RawJobStatus.FAILED, error)
            record.fail(error)
            return record

        audio_file = AudioFile(
            source_url=job.source_url,
            analysis_id=job.id,
            source_type=AudioSource(job.source_type),
            dest_dir=wav_path.parent,
        )
        data = ProcessedAudio(audio_file=audio_file, wav_path=wav_path)

        if job.annotation_path:
            data.annotation_path = Path(job.annotation_path)

        if self.pipeline_version == 2:
            processed = build_training_pipeline_v2().handle(data)
        else:
            processed = build_training_pipeline().handle(data)

        if processed.is_failed:
            await mongo_repo.update_status(job.id, RawJobStatus.FAILED, processed.error)
            record.fail(processed.error or "Pipeline failed")
            self._cleanup(wav_path)
            return record

        record.status = IngestStatus.SAVING

        if self.pipeline_version == 2 and processed.section_row:
            section = TrainingSection(**processed.section_row)
            chords = [TrainingChord(**row) for row in processed.chord_rows]
            try:
                ts_repo = TrainingSectionRepository()
                ts_repo.replace_song(job.id, section, chords)
            except Exception as e:
                error = f"Saving training_sections/chords failed: {e}"
                await mongo_repo.update_status(job.id, RawJobStatus.FAILED, error)
                record.fail(error)
                self._cleanup(wav_path)
                return record
        else:
            samples = [TrainingSample(**row) for row in processed.sample_rows]
            try:
                ts_repo_v1 = TrainingSampleRepository()
                ts_repo_v1.replace_song(job.id, samples)
            except Exception as e:
                error = f"Saving training_samples failed: {e}"
                await mongo_repo.update_status(job.id, RawJobStatus.FAILED, error)
                record.fail(error)
                self._cleanup(wav_path)
                return record

        await mongo_repo.update_status(job.id, RawJobStatus.DONE)
        record.status = IngestStatus.DONE
        self._cleanup(wav_path)
        return record

    @staticmethod
    def _cleanup(wav_path: Path) -> None:
        try:
            shutil.rmtree(wav_path.parent, ignore_errors=True)
        except Exception:
            pass
