from config import *
from model.fitness_functions import *
from model.population import Population
from selection.selection_method import SelectionMethod
from model.gen_operators import GeneticOperator
from model.experiment_helpers import ExperimentParams
from stats.run_stats import RunStats
from stats.generation_stats import GenerationStats
from output import plotting


class EvoAlgorithm:
    def __init__(self,
                 initial_population: Population,
                 experiment_params: ExperimentParams):
        self.population: Population = initial_population
        self.selection_method = experiment_params.selection_method
        self.genetic_operator = experiment_params.genetic_operator
        self.experiment_params = experiment_params

        self.gen_i = 0
        self.run_stats = RunStats(self.experiment_params)
        self.prev_gen_stats = None
        self.gen_stats_list = None
        self.has_converged = False
        
    def run(self, run_i):
        if run_i < RUNS_TO_PLOT:
            self.gen_stats_list = []

        while not self.has_converged and self.gen_i < G:
            gen_stats = self.__calculate_stats_and_evolve(run_i)

            self.has_converged = self.population.has_converged(
                has_genetic_operators=self.experiment_params.has_genetic_operator()
            )
            self.prev_gen_stats = gen_stats
            self.gen_i += 1

        gen_stats = self.__calculate_final_stats(run_i)
        self.run_stats.NI = self.gen_i
        self.run_stats.is_successful = self.__check_success(gen_stats)
        self.run_stats.has_converged = self.has_converged

        if run_i < RUNS_TO_PLOT:
            plotting.plot_generation_stats(self.population, self.experiment_params, run_i, self.gen_i)
            plotting.plot_run_stats(self.gen_stats_list, self.experiment_params, run_i)

        return self.run_stats

    def __calculate_stats_and_evolve(self, run_i):
        if run_i < RUNS_TO_PLOT and self.gen_i < DISTRIBUTIONS_TO_PLOT:
            plotting.plot_generation_stats(self.population, self.experiment_params, run_i, self.gen_i)
        
        gen_stats = GenerationStats(self.population, self.experiment_params)
        if run_i < RUNS_TO_PLOT:
            self.gen_stats_list.append(gen_stats)

        gen_stats.calculate_stats_before_selection(self.prev_gen_stats)
        self.selection_method.select(self.population)
        gen_stats.calculate_stats_after_selection()
        self.run_stats.update_stats_for_generation(gen_stats, self.gen_i)
        self.genetic_operator.apply(self.population)

        return gen_stats

    def __calculate_final_stats(self, run_i):
        if run_i < RUNS_TO_PLOT and self.gen_i < DISTRIBUTIONS_TO_PLOT:
            plotting.plot_generation_stats(self.population, self.experiment_params, run_i, self.gen_i)

        gen_stats = GenerationStats(self.population, self.experiment_params)
        if run_i < RUNS_TO_PLOT:
            self.gen_stats_list.append(gen_stats)

        gen_stats.calculate_stats_before_selection(self.prev_gen_stats)
        self.run_stats.update_final_stats(gen_stats, self.gen_i)

        return gen_stats

    def __check_success(self, gen_stats: GenerationStats):
        if not self.has_converged:
            return False

        has_genetic_operators = self.experiment_params.has_genetic_operator()

        if self.experiment_params.fitness_function.name() == 'FconstALL':
            if not has_genetic_operators:
                return True
            else:
                return self.population.most_frequent_genotype_percentage() >= 0.9
        elif self.experiment_params.fitness_function.name() in ('FHD', 'FH'):
            if not has_genetic_operators:
                return gen_stats.optimal_count >= N
            else:
                return gen_stats.optimal_count >= N * 0.9
        else:
            return self.population.found_close_to_optimal()
