from __future__ import annotations

from src.data_processing.application.mappers.harte_mapper import map_harte
from src.data_processing.domain.models.base_handler import BaseProcessingHandler
from src.data_processing.domain.models.processed_audio import ProcessedAudio


class LabelHandler(BaseProcessingHandler):

    def handle(self, data: ProcessedAudio) -> ProcessedAudio:
        if data.is_failed:
            return data

        if data.chord_timeline:
            for chord in data.chord_timeline:
                mapped = map_harte(chord["label_raw"])
                chord["label_status"] = mapped.status.value
                chord["root_id"] = mapped.root_id
                chord["quality_id"] = mapped.quality_id
                chord["ext_id"] = mapped.ext_id
        else:
            for segment in data.segments:
                if segment.chord_label is None:
                    continue
                mapped = map_harte(segment.chord_label)
                segment.label_status = mapped.status.value
                segment.root_id      = mapped.root_id
                segment.quality_id   = mapped.quality_id
                segment.ext_id       = mapped.ext_id

        return self._call_next(data)
