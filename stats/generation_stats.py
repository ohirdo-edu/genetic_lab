from config import N
from model.fitness_functions import FconstALL
from model.population import Population
from typing import Optional
from stats.selection_pressure import fisher_stat, kendall_stat
from model.experiment_helpers import ExperimentParams


# stats that are used for graphs
class GenerationStats:
    def __init__(self, population: Population, experiment_params: ExperimentParams):
        self.population = population
        self.experiment_params = experiment_params

        self.f_avg = None
        self.f_std = None
        self.f_best = None
        self.num_of_best = None
        self.optimal_count = None
        self.growth_rate = None
        self.difference = None
        self.intensity = None
        self.reproduction_rate = None
        self.loss_of_diversity = None
        self.number_of_unique_chromosomes = None

        self.optimal_loose_count = None
        self._previous_optimal_count = None

        self.Pr = None
        self.Fish = None
        self.Kend = None

    def calculate_stats_before_selection(self, prev_gen_stats: Optional['GenerationStats']):

        self.ids_before_selection = set(self.population.get_ids())
        assert len(self.ids_before_selection) == N
        self.number_of_unique_chromosomes = self.population.num_of_unique_chromosomes()

        if not isinstance(self.experiment_params.fitness_function, FconstALL):
            self.ids_before_selection_list = self.population.get_ids()
            self._previous_optimal_count = prev_gen_stats.optimal_count if prev_gen_stats else 0
            self.f_avg = self.population.get_fitness_avg()
            self.f_std = self.population.get_fitness_std()
            self.f_best = self.population.get_fitness_max()
            self.num_of_best = self.population.count_fitness_at_least(self.f_best)
            self.optimal_count = self.population.count_optimal_genotype()
            
            if not prev_gen_stats:
                self.growth_rate = 1
            else:
                num_of_prev_best = self.population.count_fitness_at_least(prev_gen_stats.f_best)
                self.growth_rate = num_of_prev_best / prev_gen_stats.num_of_best

    def calculate_stats_after_selection(self):
        ids_after_selection = set(self.population.get_ids())
        self.reproduction_rate = len(ids_after_selection) / N
        self.loss_of_diversity = len([True for id in self.ids_before_selection if id not in ids_after_selection]) / N
        self.ids_before_selection = None

        if not isinstance(self.experiment_params.fitness_function, FconstALL):
            ids_after_selection_list = self.population.get_ids()

            self.optimal_loose_count = max(0, self._previous_optimal_count - self.optimal_count)
            self.difference = self.population.get_fitness_avg() - self.f_avg

            if self.f_std == 0:
                self.intensity = 1
            else:
                self.intensity = self.difference / self.f_std

            self.Pr = self.f_best / self.f_avg
            self.Fish = fisher_stat(
                ids_before_selection=self.ids_before_selection_list,
                ids_after_selection=ids_after_selection_list,
                fitness=self.population.fitnesses,
            )
            self.Kend = kendall_stat(
                ids_before_selection=self.ids_before_selection_list,
                ids_after_selection=ids_after_selection_list,
                fitness=self.population.fitnesses,
            )
