from typing import List, Dict
from data import SONGS

def playlist_fitness(ind, target_minutes: float, w: Dict[str, float]) -> float:
    """
    Fitness function for playlist generation.
    Higher is better.

    Encourages:
    - Duration closeness to target
    - Artist/genre diversity
    - Smooth tempo & mood transitions
    - Penalizes repeated artists in short windows
    """
    tracks = [SONGS[i] for i in ind]
    total_dur = sum(t.duration_min for t in tracks)

    # 1) Duration closeness (soft): penalize deviation from target_minutes
    dur_pen = abs(total_dur - target_minutes)

    # 2) Diversity bonus: unique artists & genres
    uniq_art = len({t.artist for t in tracks})
    uniq_gen = len({t.genre for t in tracks})
    diversity = (uniq_art + 0.5 * uniq_gen) / len(tracks)

    # 3) Smooth transitions: small tempo differences & mood continuity
    tempo_pen = 0.0
    mood_bonus = 0.0
    for a, b in zip(tracks, tracks[1:]):
        tempo_pen += abs(a.tempo_bpm - b.tempo_bpm) / 200.0  # scaled
        if a.mood == b.mood:
            mood_bonus += 0.2

    # 4) Artist repeat penalty within a sliding window of 3
    repeat_pen = 0.0
    window = 3
    for i in range(len(tracks)):
        seen = {t.artist for t in tracks[max(0, i-window):i]}
        if tracks[i].artist in seen:
            repeat_pen += 0.5

    # Weighted score (larger is better)
    score = (
        w["diversity"] * diversity
        + w["mood"] * mood_bonus / max(1, len(tracks)-1)
        - w["tempo"] * tempo_pen / max(1, len(tracks)-1)
        - w["repeat"] * repeat_pen / len(tracks)
        - w["duration"] * (dur_pen / max(1.0, target_minutes))
    )
    return score


def timetable_fitness(ind, days: int, slots_per_day: int, target_moods: List[str], w: Dict[str, float]) -> float:
    """
    Fitness function for timetable generation.
    Chromosome is a permutation; we take first D*S genes as timetable in row-major order.

    Encourages:
    - Matching target mood per slot
    - Artist diversity across each day
    - Smooth daily tempo arcs
    Penalizes:
    - Artist repeats within a day
    - Abrupt tempo jumps
    """
    total_slots = days * slots_per_day
    tracks = [SONGS[i] for i in ind[:total_slots]]

    mood_match = 0.0
    tempo_smooth_pen = 0.0
    daily_diversity = 0.0
    artist_repeat_pen = 0.0

    for d in range(days):
        day_tracks = tracks[d*slots_per_day:(d+1)*slots_per_day]

        # Mood targets
        for s, t in enumerate(day_tracks):
            desired = target_moods[s % len(target_moods)]
            if t.mood == desired:
                mood_match += 1.0

        # Tempo smoothness within the day
        for a, b in zip(day_tracks, day_tracks[1:]):
            tempo_smooth_pen += abs(a.tempo_bpm - b.tempo_bpm) / 200.0

        # Diversity within the day
        daily_diversity += len({t.artist for t in day_tracks}) / max(1, len(day_tracks))

        # Artist repeat penalty within the day
        seen = set()
        for t in day_tracks:
            if t.artist in seen:
                artist_repeat_pen += 0.5
            seen.add(t.artist)

    # Normalized metrics
    mood_match_norm = mood_match / (days * slots_per_day)
    tempo_pen_norm = tempo_smooth_pen / max(1, days * (slots_per_day - 1))
    daily_div_norm = daily_diversity / days

    score = (
        w["mood"] * mood_match_norm
        + w["diversity"] * daily_div_norm
        - w["tempo"] * tempo_pen_norm
        - w["repeat"] * (artist_repeat_pen / (days * slots_per_day))
    )
    return score
