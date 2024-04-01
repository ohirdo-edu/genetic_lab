import random
from copy import copy
import numpy as np

from model.population import Population


class SelectionMethod:
    def select(self, population: Population) -> None:
        raise NotImplementedError()

    def name(self) -> str:
        return self.__class__.__name__


class SelectionMethodBase(SelectionMethod):
    def select(self, population: Population) -> None:
        population.shuffle()
        probs = self._get_probabilities(population)
        chosen = self._select_chromosomes(population, probs)
        mating_pool = np.array([copy(chr) for chr in chosen])
        population.update_chromosomes(mating_pool)

    def _get_probabilities(self, population: Population) -> list[float]:
        raise NotImplementedError()

    def _select_chromosomes(self, population: Population, probabilities: list[float]):
        raise NotImplementedError()


class RWS(SelectionMethodBase):
    def _get_probabilities(self, population: Population) -> list[float]:
        return _probabilities_from_fitness(population)

    def _select_chromosomes(self, population: Population, probabilities: list[float]):
        return _select_chromosomes_rws(population, probabilities)


class TruncatedRWS(SelectionMethodBase):
    def __init__(self, T: float):
        self.T = T

    def _get_probabilities(self, population: Population) -> list[float]:
        return _probabilities_n_best(round(self.T * len(population.chromosomes)), population)

    def _select_chromosomes(self, population: Population, probabilities: list[float]):
        return _select_chromosomes_rws(population, probabilities)

    def name(self) -> str:
        return f"{super().name()}(T={self.T})"


class SUS(SelectionMethodBase):
    def _get_probabilities(self, population: Population) -> list[float]:
        return _probabilities_from_fitness(population)

    def _select_chromosomes(self, population: Population, probabilities: list[float]):
        return _select_chromosomes_sus(population, probabilities)


class TruncatedSUS(SelectionMethodBase):
    def __init__(self, T: float):
        self.T = T

    def _get_probabilities(self, population: Population) -> list[float]:
        return _probabilities_n_best(round(self.T * len(population.chromosomes)), population)

    def _select_chromosomes(self, population: Population, probabilities: list[float]):
        return _select_chromosomes_sus(population, probabilities)

    def name(self) -> str:
        return f"{super().name()}(T={self.T})"


def _probabilities_n_best(num_of_best: int, population: Population) -> list[float]:
    fitness_list = population.fitnesses
    population_size = len(fitness_list)
    best_indexes = set(sorted(range(population_size), key=lambda i: fitness_list[i], reverse=True)[:num_of_best])

    return [(1 / num_of_best if i in best_indexes else 0) for i in range(population_size)]


def _probabilities_from_fitness(population: Population) -> list[float]:
    fitness_list = population.fitnesses
    population_size = len(fitness_list)
    fitness_sum = sum(fitness_list)

    if fitness_sum == 0:
        fitness_list = [0.0001 for _ in fitness_list]
        fitness_sum = 0.0001 * population_size

    return [fitness / fitness_sum for fitness in fitness_list]


def _select_chromosomes_rws(population: Population, probabilities: list[float]):
    population_size = len(population.fitnesses)
    return np.random.choice(population.chromosomes, size=population_size, p=probabilities)


def _select_chromosomes_sus(population: Population, probabilities: list[float]):
    population_size = len(population.fitnesses)
    distance = 1 / population_size
    start = np.random.uniform(0, distance)
    points = [start + i * distance for i in range(population_size)]

    chosen_indexes = []
    for p in points:
        i = 0
        sum_ = probabilities[i]
        while sum_ < p:
            i += 1
            sum_ += probabilities[i]
        chosen_indexes.append(i)

    return np.array([population.chromosomes[i] for i in chosen_indexes])
