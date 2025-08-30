from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class Song:
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    tempo_bpm: int
    duration_min: float

# Demo catalogue
SONGS: List[Song] = [
    Song(1, "Aurora Dawn", "Nyra", "pop", "happy", 120, 3.2),
    Song(2, "Midnight Lane", "Nyra", "pop", "calm", 88, 3.8),
    # ... (rest of songs)
]

MOODS = ["calm", "happy", "energetic", "sad"]
