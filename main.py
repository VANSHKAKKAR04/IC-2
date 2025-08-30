#!/usr/bin/env python3

from __future__ import annotations
import argparse
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Callable, Dict, Any, Sequence

# -----------------------------
# Domain model
# -----------------------------
@dataclass(frozen=True)
class Song:
    id: int
    title: str
    artist: str
    genre: str
    mood: str         # e.g., calm, happy, energetic, sad
    tempo_bpm: int
    duration_min: float

# Small demo catalogue (feel free to replace)
SONGS: List[Song] = [
    Song(1, "Aurora Dawn", "Nyra", "pop", "happy", 120, 3.2),
    Song(2, "Midnight Lane", "Nyra", "pop", "calm", 88, 3.8),
    Song(3, "Iron Skies", "Volt", "rock", "energetic", 142, 4.1),
    Song(4, "Paper Boats", "Milo", "indie", "calm", 95, 3.6),
    Song(5, "Firefly", "Zaya", "edm", "energetic", 128, 3.4),
    Song(6, "Slow River", "Milo", "indie", "sad", 78, 4.0),
    Song(7, "City Lights", "Zaya", "edm", "happy", 126, 3.3),
    Song(8, "Sunset Code", "Volt", "rock", "happy", 136, 3.9),
    Song(9, "Nebula Love", "Arin", "rnb", "calm", 92, 3.7),
    Song(10, "Blue Monday", "Arin", "rnb", "sad", 84, 4.2),
    Song(11, "Drift", "Kei", "lofi", "calm", 72, 2.8),
    Song(12, "Uptick", "Kei", "lofi", "happy", 78, 2.6),
    Song(13, "Thunder Run", "Volt", "rock", "energetic", 150, 4.0),
    Song(14, "Glass Garden", "Nyra", "pop", "sad", 100, 3.5),
    Song(15, "Wild Circuit", "Zaya", "edm", "energetic", 132, 3.1),
    Song(16, "Rain Letters", "Milo", "indie", "sad", 82, 3.9),
    Song(17, "Morning Chai", "Kei", "lofi", "happy", 76, 2.5),
    Song(18, "Orbit", "Arin", "rnb", "happy", 96, 3.0),
    Song(19, "Pulse", "Zaya", "edm", "energetic", 130, 3.0),
    Song(20, "Dune Walk", "Milo", "indie", "calm", 98, 3.2),
    Song(21, "Afterglow", "Nyra", "pop", "happy", 118, 3.4),
    Song(22, "Stonewave", "Volt", "rock", "sad", 110, 4.3),
    Song(23, "Echo Bay", "Kei", "lofi", "calm", 70, 2.7),
    Song(24, "Skytrail", "Arin", "rnb", "energetic", 122, 3.5),
    Song(25, "Nimbus", "Zaya", "edm", "happy", 124, 3.2),
]

MOODS = ["calm", "happy", "energetic", "sad"]

# -----------------------------
# GA primitives
# -----------------------------
Individual = List[int]  # permutation of song indices into SONGS
Population = List[Individual]


def seeded(seed: int | None):
    if seed is not None:
        random.seed(seed)


def make_initial_population(n_pop: int, gene_pool: Sequence[int], chrom_len: int) -> Population:
    pop: Population = []
    for _ in range(n_pop):
        ind = random.sample(gene_pool, k=chrom_len)
        pop.append(ind)
    return pop


# -------- Selection operators --------

def roulette_selection(pop: Population, fitnesses: List[float], k: int) -> Population:
    # Convert to positive probabilities
    min_fit = min(fitnesses)
    offset = -min_fit + 1e-9 if min_fit < 0 else 1e-9
    weights = [f + offset for f in fitnesses]
    total = sum(weights)
    probs = [w / total for w in weights]
    chosen = random.choices(population=list(range(len(pop))), weights=probs, k=k)
    return [pop[i][:] for i in chosen]


def tournament_selection(pop: Population, fitnesses: List[float], k: int, tsize: int = 3) -> Population:
    chosen = []
    for _ in range(k):
        contestants = random.sample(list(range(len(pop))), tsize)
        best = max(contestants, key=lambda i: fitnesses[i])
        chosen.append(pop[best][:])
    return chosen


def rank_selection(pop: Population, fitnesses: List[float], k: int) -> Population:
    ranked = sorted(range(len(pop)), key=lambda i: fitnesses[i], reverse=True)
    # Linear ranking probabilities
    n = len(pop)
    weights = [n - r for r in range(n)]
    total = sum(weights)
    probs = [w / total for w in weights]
    chosen_idx = random.choices(ranked, weights=probs, k=k)
    return [pop[i][:] for i in chosen_idx]


SELECTIONS: Dict[str, Callable[[Population, List[float], int], Population]] = {
    "roulette": roulette_selection,
    "tournament": tournament_selection,
    "rank": rank_selection,
}

# -------- Crossover operators --------

def one_point_crossover(p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
    cut = random.randint(1, len(p1) - 1)
    c1 = p1[:cut] + [g for g in p2 if g not in p1[:cut]]
    c2 = p2[:cut] + [g for g in p1 if g not in p2[:cut]]
    return c1, c2


def two_point_crossover(p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
    a, b = sorted(random.sample(range(len(p1)), 2))
    def mix(a1, a2):
        child = [-1] * len(p1)
        child[a:b+1] = a1[a:b+1]
        fill = [g for g in a2 if g not in child]
        idx = 0
        for i in range(len(child)):
            if child[i] == -1:
                child[i] = fill[idx]
                idx += 1
        return child
    return mix(p1, p2), mix(p2, p1)


def uniform_crossover(p1: Individual, p2: Individual, swap_prob: float = 0.5) -> Tuple[Individual, Individual]:
    c1 = p1[:]
    c2 = p2[:]
    for i in range(len(p1)):
        if random.random() < swap_prob:
            # swap positions i by mapping values uniquely (repair after)
            v1, v2 = c1[i], c2[i]
            j1 = c1.index(v2)
            j2 = c2.index(v1)
            c1[i], c1[j1] = c1[j1], c1[i]
            c2[i], c2[j2] = c2[j2], c2[i]
    return c1, c2


def order1_crossover(p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
    # Classic OX for permutations
    a, b = sorted(random.sample(range(len(p1)), 2))
    def ox(pa, pb):
        child = [-1] * len(pa)
        child[a:b+1] = pa[a:b+1]
        fill = [g for g in pb if g not in child]
        idx = 0
        for i in range(len(child)):
            if child[i] == -1:
                child[i] = fill[idx]
                idx += 1
        return child
    return ox(p1, p2), ox(p2, p1)


CROSSOVERS: Dict[str, Callable[[Individual, Individual], Tuple[Individual, Individual]]] = {
    "one_point": one_point_crossover,
    "two_point": two_point_crossover,
    "uniform": uniform_crossover,
    "order1": order1_crossover,
}

# -------- Mutation operators --------

def mut_swap(ind: Individual, rate: float) -> None:
    if random.random() < rate:
        i, j = random.sample(range(len(ind)), 2)
        ind[i], ind[j] = ind[j], ind[i]


def mut_scramble(ind: Individual, rate: float) -> None:
    if random.random() < rate:
        a, b = sorted(random.sample(range(len(ind)), 2))
        seg = ind[a:b+1]
        random.shuffle(seg)
        ind[a:b+1] = seg


def mut_inversion(ind: Individual, rate: float) -> None:
    if random.random() < rate:
        a, b = sorted(random.sample(range(len(ind)), 2))
        ind[a:b+1] = reversed(ind[a:b+1])


def mut_replace(ind: Individual, rate: float, gene_pool: Sequence[int]) -> None:
    if random.random() < rate:
        i = random.randrange(len(ind))
        remaining = [g for g in gene_pool if g not in ind or g == ind[i]]
        ind[i] = random.choice(remaining)


MUTATIONS: Dict[str, Callable[..., None]] = {
    "swap": mut_swap,
    "scramble": mut_scramble,
    "inversion": mut_inversion,
    "replace": mut_replace,
}

# -----------------------------
# Fitness functions
# -----------------------------

def playlist_fitness(ind: Individual, target_minutes: float, w: Dict[str, float]) -> float:
    """Higher is better."""
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

    # Normalize rough score (larger is better)
    score = (
        w["diversity"] * diversity
        + w["mood"] * mood_bonus / max(1, len(tracks)-1)
        - w["tempo"] * tempo_pen / max(1, len(tracks)-1)
        - w["repeat"] * repeat_pen / len(tracks)
        - w["duration"] * (dur_pen / max(1.0, target_minutes))
    )
    return score


def timetable_fitness(ind: Individual, days: int, slots_per_day: int, target_moods: List[str], w: Dict[str, float]) -> float:
    """Chromosome is a permutation; we take first D*S genes as timetable in row-major.
    Fitness encourages matching target mood per slot, diversity across a day, and smooth daily tempo arcs.
    """
    total_slots = days * slots_per_day
    tracks = [SONGS[i] for i in ind[:total_slots]]

    mood_match = 0.0
    tempo_smooth_pen = 0.0
    daily_diversity = 0.0
    artist_repeat_pen = 0.0

    for d in range(days):
        day_tracks = tracks[d*slots_per_day:(d+1)*slots_per_day]
        # mood targets
        for s, t in enumerate(day_tracks):
            desired = target_moods[s % len(target_moods)]
            if t.mood == desired:
                mood_match += 1.0
        # tempo smoothness within the day
        for a, b in zip(day_tracks, day_tracks[1:]):
            tempo_smooth_pen += abs(a.tempo_bpm - b.tempo_bpm) / 200.0
        # diversity within the day
        daily_diversity += len({t.artist for t in day_tracks}) / max(1, len(day_tracks))
        # artist repeat within day
        seen = set()
        for t in day_tracks:
            if t.artist in seen:
                artist_repeat_pen += 0.5
            seen.add(t.artist)

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

# -----------------------------
# GA loop
# -----------------------------

def evolve(
    fitness_fn: Callable[[Individual], float],
    gene_pool: Sequence[int],
    chrom_len: int,
    pop_size: int = 100,
    gens: int = 200,
    selection: str = "tournament",
    crossover: str = "order1",
    mutation: str = "swap",
    cx_prob: float = 0.9,
    mut_rate: float = 0.2,
    elitism: int = 2,
    seed: int | None = 42,
) -> Tuple[Individual, float, List[float]]:
    seeded(seed)
    pop = make_initial_population(pop_size, gene_pool, chrom_len)
    best_hist: List[float] = []

    for g in range(gens):
        fitnesses = [fitness_fn(ind) for ind in pop]
        # Elites
        elite_idx = sorted(range(len(pop)), key=lambda i: fitnesses[i], reverse=True)[:elitism]
        elites = [pop[i][:] for i in elite_idx]

        # Selection
        sel_fn = SELECTIONS[selection]
        parents = sel_fn(pop, fitnesses, k=pop_size - elitism)

        # Crossover
        next_pop: Population = elites[:]
        random.shuffle(parents)
        for i in range(0, len(parents)-1, 2):
            p1, p2 = parents[i], parents[i+1]
            if random.random() < cx_prob:
                c1, c2 = CROSSOVERS[crossover](p1, p2)
            else:
                c1, c2 = p1[:], p2[:]
            next_pop.extend([c1, c2])
        if len(next_pop) < pop_size:
            next_pop.append(parents[-1][:])

        # Mutation
        for ind in next_pop[elitism:]:
            if mutation == "replace":
                MUTATIONS[mutation](ind, mut_rate, gene_pool)
            else:
                MUTATIONS[mutation](ind, mut_rate)

        pop = next_pop[:pop_size]
        best = max(fitnesses)
        best_hist.append(best)
    # Final best
    fitnesses = [fitness_fn(ind) for ind in pop]
    best_i = max(range(len(pop)), key=lambda i: fitnesses[i])
    return pop[best_i], fitnesses[best_i], best_hist


# -----------------------------
# CLI wiring
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="GA for musical playlist & timetable generation")
    parser.add_argument("--mode", choices=["playlist", "timetable"], required=True)
    parser.add_argument("--pop-size", type=int, default=120)
    parser.add_argument("--gens", type=int, default=150)
    parser.add_argument("--selection", choices=list(SELECTIONS.keys()), default="tournament")
    parser.add_argument("--crossover", choices=list(CROSSOVERS.keys()), default="order1")
    parser.add_argument("--mutation", choices=list(MUTATIONS.keys()), default="swap")
    parser.add_argument("--cx-prob", type=float, default=0.9)
    parser.add_argument("--mut-rate", type=float, default=0.2)
    parser.add_argument("--elitism", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    # Playlist-specific
    parser.add_argument("--playlist-size", type=int, default=15)
    parser.add_argument("--target-minutes", type=float, default=50.0)

    # Timetable-specific
    parser.add_argument("--days", type=int, default=5)
    parser.add_argument("--slots-per-day", type=int, default=4)
    parser.add_argument("--targets", nargs="*", default=["calm", "happy", "energetic", "calm"],
                        help="Target moods per slot (cycled across days)")

    # Fitness weights
    parser.add_argument("--w-duration", type=float, default=1.0)
    parser.add_argument("--w-diversity", type=float, default=1.5)
    parser.add_argument("--w-mood", type=float, default=2.0)
    parser.add_argument("--w-tempo", type=float, default=1.0)
    parser.add_argument("--w-repeat", type=float, default=1.0)

    args = parser.parse_args()

    weights = {
        "duration": args.w_duration,
        "diversity": args.w_diversity,
        "mood": args.w_mood,
        "tempo": args.w_tempo,
        "repeat": args.w_repeat,
    }

    gene_pool = list(range(len(SONGS)))

    if args.mode == "playlist":
        chrom_len = min(args.playlist_size, len(SONGS))
        def fit(ind: Individual) -> float:
            return playlist_fitness(ind, target_minutes=args.target_minutes, w=weights)
        best, score, hist = evolve(
            fitness_fn=fit,
            gene_pool=gene_pool,
            chrom_len=chrom_len,
            pop_size=args.pop_size,
            gens=args.gens,
            selection=args.selection,
            crossover=args.crossover,
            mutation=args.mutation,
            cx_prob=args.cx_prob,
            mut_rate=args.mut_rate,
            elitism=args.elitism,
            seed=args.seed,
        )
        print("Best playlist score:", round(score, 4))
        total_min = sum(SONGS[i].duration_min for i in best)
        print(f"Total duration: {total_min:.1f} min (target {args.target_minutes:.1f})\n")
        print("ORDERED PLAYLIST:")
        for pos, idx in enumerate(best, 1):
            s = SONGS[idx]
            print(f"{pos:02d}. {s.title:15s} | {s.artist:6s} | {s.genre:5s} | {s.mood:9s} | {s.tempo_bpm:3d} bpm | {s.duration_min:.1f} min")

    else:  # timetable
        total_slots = args.days * args.slots_per_day
        chrom_len = min(total_slots, len(SONGS))
        target_moods = args.targets
        def fit(ind: Individual) -> float:
            return timetable_fitness(ind, days=args.days, slots_per_day=args.slots_per_day, target_moods=target_moods, w=weights)
        best, score, hist = evolve(
            fitness_fn=fit,
            gene_pool=gene_pool,
            chrom_len=chrom_len,
            pop_size=args.pop_size,
            gens=args.gens,
            selection=args.selection,
            crossover=args.crossover,
            mutation=args.mutation,
            cx_prob=args.cx_prob,
            mut_rate=args.mut_rate,
            elitism=args.elitism,
            seed=args.seed,
        )
        print("Best timetable score:", round(score, 4))
        print()
        print("WEEKLY TIMETABLE (row = day, col = slot)")
        for d in range(args.days):
            row = []
            for s in range(args.slots_per_day):
                idx = best[d*args.slots_per_day + s]
                t = SONGS[idx]
                row.append(f"{t.title} [{t.mood}/{t.tempo_bpm}]")
            print(f"Day {d+1:02d}: ", " | ".join(row))
        print("\nTarget moods per slot:", target_moods)


if __name__ == "__main__":
    main()
