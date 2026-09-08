"""Repository for SongAnalysis — extends BaseRepository with domain mapping and URL cache lookup."""
from __future__ import annotations
from sqlalchemy import select

from src.shared.domain.processing_result import ProcessingResult
from src.shared.infrastructure.postgres.base_repo import BaseRepository
from src.song_analysis.domain.models.song_analysis import SongAnalysis
from src.song_analysis.infrastructure.orm.song_analysis_orm import SongAnalysisORM


class SongAnalysisRepository(BaseRepository[SongAnalysisORM]):
    """Concrete repository for SongAnalysis.

    Inherits save, get_by_id, delete_by_id, list_all from BaseRepository.
    Adds domain mapping (save_domain, get_domain_by_id) and URL cache lookup.
    """

    model = SongAnalysisORM

    async def save_domain(self, entity: SongAnalysis) -> None:
        """Upsert SongAnalysis via session.merge() — atomic, no race condition.

        merge() checks identity map by PK: inserts if new, updates if existing.
        """
        await self.session.merge(SongAnalysisORM.from_domain(entity))
        await self.session.flush()

    async def get_domain_by_id(self, analysis_id: str) -> ProcessingResult[SongAnalysis]:
        """Fetch domain entity by UUID. Returns Err if not found."""
        row = await self.get_by_id(analysis_id)
        if row is None: return ProcessingResult.err(f"SongAnalysis not found: {analysis_id}")
        
        return ProcessingResult.ok(row.to_domain())

    async def get_by_url(self, source_url: str) -> ProcessingResult[SongAnalysis]:
        """Fetch a completed analysis by source URL (cache check before re-downloading).

        Used in DAG to skip pipeline if the song was already processed.

        Example:
            cached = await repo.get_by_url(youtube_url)
            if cached.is_ok:
                return cached.unwrap()  # skip download + processing
        """
        stmt = select(SongAnalysisORM).where(
            SongAnalysisORM.source_url == source_url,
            SongAnalysisORM.status == "done",
        )
        result = await self.session.execute(stmt)
        row = result.scalar_one_or_none()
        if row is None: return ProcessingResult.err(f"No cached analysis for: {source_url}")
        
        return ProcessingResult.ok(row.to_domain())
