"""Shared types. The JSON writer is the only place these become dicts."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Event:
    class_name: str                      # one of the 11; NEVER "normal"
    start_time_sec: Optional[float]      # None at level 1, required and >=0 at 2/3
    end_time_sec: Optional[float]        # None at level 1, must be > start
    score: float = 0.0                   # confidence; preserves ranking and L1 tie-breaking
    explanation: Optional[str] = None    # 20-500 chars or omitted entirely


@dataclass
class VideoPred:
    video_id: str
    level: int
    events: list[Event] = field(default_factory=list)
    frames_processed: int = 0
    chunks_processed: int = 0
    end_to_end_internal_time_ms: float = 0.0

    def to_dict(self) -> dict:
        """Emit submission JSON for one video.

        Invariants enforced here (violating any one gets the file rejected):
          - level 1 -> both timestamps serialize as null
          - level 2/3 -> both timestamps present, end > start
          - `score` is finite and preserved for ranking; `explanation` omitted when invalid
          - runtime_metadata always present
        """
        events = []
        for event in self.events:
            if event.class_name == "normal":
                raise ValueError(f"{self.video_id}: class_name 'normal' must be events: []")
            if self.level == 1:
                if event.start_time_sec is not None or event.end_time_sec is not None:
                    raise ValueError(f"{self.video_id}: level 1 event timestamps must be None")
            elif (event.start_time_sec is None or event.end_time_sec is None or
                  event.end_time_sec <= event.start_time_sec):
                raise ValueError(f"{self.video_id}: invalid level {self.level} event interval")
            item = {
                "class_name": event.class_name,
                "start_time_sec": event.start_time_sec,
                "end_time_sec": event.end_time_sec,
                "score": float(event.score),
            }
            if event.explanation is not None and 20 <= len(event.explanation) <= 500:
                item["explanation"] = event.explanation
            events.append(item)
        return {
            "video_id": self.video_id,
            "events": events,
            "runtime_metadata": {
                "frames_processed": self.frames_processed,
                "chunks_processed": self.chunks_processed,
                "end_to_end_internal_time_ms": self.end_to_end_internal_time_ms,
            },
        }
