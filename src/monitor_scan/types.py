from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)


@dataclass(frozen=True)
class PersonDetection:
    box: BoundingBox
    confidence: float


@dataclass(frozen=True)
class DetectionEvent:
    video_name: str
    timestamp: str
    confidence: float
    snapshot_path: str
