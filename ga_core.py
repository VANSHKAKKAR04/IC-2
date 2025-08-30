import random
from typing import List, Sequence, Callable, Tuple

Individual = List[int]
Population = List[Individual]

def seeded(seed: int | None):
    if seed is not None:
        random.seed(seed)

def make_initial_population(n_pop: int, gene_pool: Sequence[int], chrom_len: int) -> Population:
    return [random.sample(gene_pool, k=chrom_len) for _ in range(n_pop)]

def evolve(
    fitness_fn: Callable[[Individual], float],
    gene_pool: Sequence[int],
    chrom_len: int,
    pop_size: int,
    gens: int,
    selection,
    crossover,
    mutation,
    cx_prob: float = 0.9,
    mut_rate: float = 0.2,
    elitism: int = 2,
    seed: int | None = 42,
):
    seeded(seed)
    pop = make_initial_population(pop_size, gene_pool, chrom_len)
    best_hist: List[float] = []

    for _ in range(gens):
        fitnesses = [fitness_fn(ind) for ind in pop]
        elite_idx = sorted(range(len(pop)), key=lambda i: fitnesses[i], reverse=True)[:elitism]
        elites = [pop[i][:] for i in elite_idx]

        parents = selection(pop, fitnesses, k=pop_size - elitism)

        next_pop: Population = elites[:]
        random.shuffle(parents)
        for i in range(0, len(parents)-1, 2):
            p1, p2 = parents[i], parents[i+1]
            if random.random() < cx_prob:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]
            next_pop.extend([c1, c2])
        if len(next_pop) < pop_size:
            next_pop.append(parents[-1][:])

        for ind in next_pop[elitism:]:
            mutation(ind, mut_rate)

        pop = next_pop[:pop_size]
        best_hist.append(max(fitnesses))

    fitnesses = [fitness_fn(ind) for ind in pop]
    best_i = max(range(len(pop)), key=lambda i: fitnesses[i])
    return pop[best_i], fitnesses[best_i], best_hist
