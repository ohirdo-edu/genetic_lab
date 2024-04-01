from dataclasses import dataclass


class PopulationInitialization:
    def number_of_optimal_individuals(self, population_size: int) -> int:
        raise NotImplementedError


@dataclass
class ConstantPopulationInitialization(PopulationInitialization):
    constant_optimal_size: int

    def number_of_optimal_individuals(self, population_size: int) -> int:
        return self.constant_optimal_size

    def __str__(self):
        return f"={self.constant_optimal_size}"


@dataclass
class PercentagePopulationInitialization(PopulationInitialization):
    optimal_size_percent: int

    def number_of_optimal_individuals(self, population_size: int) -> int:
        return round(population_size * self.optimal_size_percent / 100)

    def __str__(self):
        return f"{self.optimal_size_percent}%"
