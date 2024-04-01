from dataclasses import dataclass

from .fitness_functions import FitnessFunc, FconstALL
from .gen_operators import GeneticOperator, BlankGenOperator
from selection.selection_method import SelectionMethod
from .population_initialization import PopulationInitialization


@dataclass
class ExperimentParams:
    fitness_function: FitnessFunc
    genetic_operator: GeneticOperator
    selection_method: SelectionMethod
    population_initialization: PopulationInitialization

    def should_calculate_convergence_stats(self) -> bool:
        return not isinstance(self.fitness_function, FconstALL)

    def should_calculate_pressure_stats(self) -> bool:
        return not isinstance(self.fitness_function, FconstALL)

    def has_genetic_operator(self) -> bool:
        return not isinstance(self.genetic_operator, BlankGenOperator)

    def __str__(self):
        return f"{self.fitness_function.name()}|{self.genetic_operator.name()}|{self.selection_method.name()}|{self.population_initialization}"
