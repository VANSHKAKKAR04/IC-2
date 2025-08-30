from data import SONGS
from ga_core import evolve
from operators import SELECTIONS, CROSSOVERS, MUTATIONS
from fitness import playlist_fitness, timetable_fitness
from cli import get_args


def get_mutation_func(name, gene_pool):
    """Return correct mutation function, handling replace separately."""
    if name not in MUTATIONS:
        raise ValueError(f"Unknown mutation: {name}")
    if name == "replace":
        return lambda ind, rate: MUTATIONS[name](ind, rate, gene_pool)
    return MUTATIONS[name]


def main():
    args = get_args()

    # fitness weights
    weights = {
        "duration": args.w_duration,
        "diversity": args.w_diversity,
        "mood": args.w_mood,
        "tempo": args.w_tempo,
        "repeat": args.w_repeat,
    }
    gene_pool = list(range(len(SONGS)))

    # resolve operators
    if args.selection not in SELECTIONS:
        raise ValueError(f"Unknown selection operator: {args.selection}")
    if args.crossover not in CROSSOVERS:
        raise ValueError(f"Unknown crossover operator: {args.crossover}")

    selection = SELECTIONS[args.selection]
    crossover = CROSSOVERS[args.crossover]
    mutation = get_mutation_func(args.mutation, gene_pool)

    if args.mode == "playlist":
        chrom_len = min(args.playlist_size, len(SONGS))
        fitness_fn = lambda ind: playlist_fitness(ind, target_minutes=args.target_minutes, w=weights)

        best, score, _ = evolve(
            fitness_fn, gene_pool, chrom_len, args.pop_size, args.gens,
            selection, crossover, mutation,
            args.cx_prob, args.mut_rate, args.elitism, args.seed
        )

        print("\n=== Best Playlist ===")
        print("Score:", round(score, 4))
        for pos, idx in enumerate(best, 1):
            s = SONGS[idx]
            print(f"{pos:02d}. {s.title} | {s.artist} | {s.genre} | {s.mood} | {s.tempo_bpm} bpm | {s.duration_min} min")

    elif args.mode == "timetable":
        chrom_len = min(args.days * args.slots_per_day, len(SONGS))
        fitness_fn = lambda ind: timetable_fitness(ind, args.days, args.slots_per_day, args.targets, w=weights)

        best, score, _ = evolve(
            fitness_fn, gene_pool, chrom_len, args.pop_size, args.gens,
            selection, crossover, mutation,
            args.cx_prob, args.mut_rate, args.elitism, args.seed
        )

        print("\n=== Best Timetable ===")
        print("Score:", round(score, 4))
        for d in range(args.days):
            row = [SONGS[best[d * args.slots_per_day + s]].title for s in range(args.slots_per_day)]
            print(f"Day {d+1:02d}: " + " | ".join(row))

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
