import random
from typing import List, Tuple

Individual = List[int]
Population = List[Individual]

# -------- Selection --------
def roulette_selection(pop: Population, fitnesses: List[float], k: int) -> Population:
    """Roulette wheel selection"""
    total_fitness = sum(fitnesses)
    selected = []
    for _ in range(k):
        pick = random.uniform(0, total_fitness)
        current = 0
        for ind, fit in zip(pop, fitnesses):
            current += fit
            if current > pick:
                selected.append(ind[:])  # copy
                break
    return selected


def tournament_selection(pop: Population, fitnesses: List[float], k: int, tsize: int = 3) -> Population:
    """Tournament selection"""
    selected = []
    for _ in range(k):
        participants = random.sample(list(zip(pop, fitnesses)), tsize)
        winner = max(participants, key=lambda x: x[1])
        selected.append(winner[0][:])
    return selected


def rank_selection(pop: Population, fitnesses: List[float], k: int) -> Population:
    """Rank-based selection"""
    sorted_pop = sorted(zip(pop, fitnesses), key=lambda x: x[1])
    total_rank = sum(range(1, len(pop) + 1))
    selected = []
    for _ in range(k):
        pick = random.uniform(0, total_rank)
        current = 0
        for rank, (ind, _) in enumerate(sorted_pop, start=1):
            current += rank
            if current >= pick:
                selected.append(ind[:])
                break
    return selected


SELECTIONS = {
    "roulette": roulette_selection,
    "tournament": tournament_selection,
    "rank": rank_selection,
}


# -------- Crossover --------
def one_point_crossover(p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
    """One-point crossover"""
    point = random.randint(1, len(p1) - 1)
    c1 = p1[:point] + p2[point:]
    c2 = p2[:point] + p1[point:]
    return c1, c2


def two_point_crossover(p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
    """Two-point crossover"""
    pt1, pt2 = sorted(random.sample(range(len(p1)), 2))
    c1 = p1[:pt1] + p2[pt1:pt2] + p1[pt2:]
    c2 = p2[:pt1] + p1[pt1:pt2] + p2[pt2:]
    return c1, c2


def uniform_crossover(p1: Individual, p2: Individual, swap_prob: float = 0.5) -> Tuple[Individual, Individual]:
    """Uniform crossover"""
    c1, c2 = [], []
    for g1, g2 in zip(p1, p2):
        if random.random() < swap_prob:
            c1.append(g2)
            c2.append(g1)
        else:
            c1.append(g1)
            c2.append(g2)
    return c1, c2


def order1_crossover(p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
    """Order1 crossover (OX1) for permutations"""
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))

    def ox(parent1, parent2):
        child = [None] * size
        child[a:b] = parent1[a:b]
        pos = b
        for gene in parent2:
            if gene not in child:
                if pos >= size:
                    pos = 0
                child[pos] = gene
                pos += 1
        return child

    return ox(p1, p2), ox(p2, p1)


CROSSOVERS = {
    "one_point": one_point_crossover,
    "two_point": two_point_crossover,
    "uniform": uniform_crossover,
    "order1": order1_crossover,
}


# -------- Mutation --------
def mut_swap(ind: Individual, rate: float) -> Individual:
    """Swap mutation"""
    mutant = ind[:]
    if random.random() < rate:
        i, j = random.sample(range(len(mutant)), 2)
        mutant[i], mutant[j] = mutant[j], mutant[i]
    return mutant


def mut_scramble(ind: Individual, rate: float) -> Individual:
    """Scramble mutation"""
    mutant = ind[:]
    if random.random() < rate:
        i, j = sorted(random.sample(range(len(mutant)), 2))
        subset = mutant[i:j]
        random.shuffle(subset)
        mutant[i:j] = subset
    return mutant


def mut_inversion(ind: Individual, rate: float) -> Individual:
    """Inversion mutation"""
    mutant = ind[:]
    if random.random() < rate:
        i, j = sorted(random.sample(range(len(mutant)), 2))
        mutant[i:j] = reversed(mutant[i:j])
    return mutant


def mut_replace(ind: Individual, rate: float, gene_pool: List[int]) -> Individual:
    """Replace mutation: replace one gene with a random one"""
    mutant = ind[:]
    if random.random() < rate:
        i = random.randrange(len(mutant))
        mutant[i] = random.choice(gene_pool)
    return mutant


MUTATIONS = {
    "swap": mut_swap,
    "scramble": mut_scramble,
    "inversion": mut_inversion,
    "replace": mut_replace,
}
