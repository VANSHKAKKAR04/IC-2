import argparse
from operators import SELECTIONS, CROSSOVERS, MUTATIONS

def get_args():
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

    parser.add_argument("--playlist-size", type=int, default=15)
    parser.add_argument("--target-minutes", type=float, default=50.0)

    parser.add_argument("--days", type=int, default=5)
    parser.add_argument("--slots-per-day", type=int, default=4)
    parser.add_argument("--targets", nargs="*", default=["calm", "happy", "energetic", "calm"])

    parser.add_argument("--w-duration", type=float, default=1.0)
    parser.add_argument("--w-diversity", type=float, default=1.5)
    parser.add_argument("--w-mood", type=float, default=2.0)
    parser.add_argument("--w-tempo", type=float, default=1.0)
    parser.add_argument("--w-repeat", type=float, default=1.0)

    return parser.parse_args()
