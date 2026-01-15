from dataclasses import dataclass


@dataclass
class Segment:
    """Information about a high-density period (segment)

    Attributes:
        label: Segment ID (sequential number starting from 0)
        start_time: Start time of the segment (minimum value of change points)
        end_time: End time of the segment (maximum value of change points)
    """
    label: int
    start_time: int
    end_time: int
