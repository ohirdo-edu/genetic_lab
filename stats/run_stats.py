from config import N
from model.experiment_helpers import ExperimentParams
from stats.generation_stats import GenerationStats
from model.fitness_functions import FconstALL
import math


class RunStats:
    def __init__(self, experiment_params: ExperimentParams):
        self.experiment_params = experiment_params

        self.NI = None
        self.F_found = None
        self.F_avg = None
        self.is_successful = False
        self.has_converged = False

        self.NI_loose = 0
        self.optSaved_NI_loose = 0
        self.Num_loose = 0

        # Reproduction Rate
        self.RR_start = None
        self.RR_fin = None
        self.RR_min = None
        self.NI_RR_min = None
        self.RR_max = None
        self.NI_RR_max = None
        self.RR_avg = None

        # Loss of Diversity
        self.Teta_start = None
        self.Teta_fin = None
        self.Teta_min = None
        self.NI_Teta_min = None
        self.Teta_max = None
        self.NI_Teta_max = None
        self.Teta_avg = None

        self.unique_X_start = None
        self.unique_X_fin = None

        # Selection Intensity
        self.I_start = None
        self.I_min = None
        self.NI_I_min = None
        self.I_max = None
        self.NI_I_max = None
        self.I_avg = None

        # Selection Difference
        self.s_start = None
        self.s_min = None
        self.NI_s_min = None
        self.s_max = None
        self.NI_s_max = None
        self.s_avg = None

        # Growth Rate
        self.GR_start = None
        self.GR_early = None
        self.GR_late = None
        self.NI_GR_late = None
        self.GR_avg = None

        self.Pr_start = None
        self.Pr_min = None
        self.NI_Pr_min = None
        self.Pr_max = None
        self.NI_Pr_max = None
        self.Pr_avg = None

        self.Fish_start = None
        self.Fish_min = None
        self.NI_Fish_min = None
        self.Fish_max = None
        self.NI_Fish_max = None
        self.Fish_avg = None

        self.Kend_start = None
        self.Kend_min = None
        self.NI_Kend_min = None
        self.Kend_max = None
        self.NI_Kend_max = None
        self.Kend_avg = None


    def update_stats_for_generation(self, gen_stats: GenerationStats, gen_i):
        self.RR_fin = gen_stats.reproduction_rate
        self.Teta_fin = gen_stats.loss_of_diversity
        self.unique_X_fin = gen_stats.number_of_unique_chromosomes

        # Reproduction Rate
        if self.RR_min is None or gen_stats.reproduction_rate < self.RR_min:
            self.RR_min = gen_stats.reproduction_rate
            self.NI_RR_min = gen_i
        if self.RR_max is None or gen_stats.reproduction_rate > self.RR_max:
            self.RR_max = gen_stats.reproduction_rate
            self.NI_RR_max = gen_i
        if self.RR_avg is None:
            self.RR_avg = gen_stats.reproduction_rate
        else:
            self.RR_avg = (self.RR_avg * (gen_i - 1) + gen_stats.reproduction_rate) / gen_i

        # Loss of Diversity
        if gen_i == 0:
            self.RR_start = gen_stats.reproduction_rate
            self.Teta_start = gen_stats.loss_of_diversity
            self.unique_X_start = gen_stats.number_of_unique_chromosomes
        if self.Teta_min is None or gen_stats.loss_of_diversity < self.Teta_min:
            self.Teta_min = gen_stats.loss_of_diversity
            self.NI_Teta_min = gen_i
        if self.Teta_max is None or gen_stats.loss_of_diversity > self.Teta_max:
            self.Teta_max = gen_stats.loss_of_diversity
            self.NI_Teta_max = gen_i
        if self.Teta_avg is None:
            self.Teta_avg = gen_stats.loss_of_diversity
        else:
            self.Teta_avg = (self.Teta_avg * (gen_i - 1) + gen_stats.loss_of_diversity) / gen_i

        if not isinstance(self.experiment_params.fitness_function, FconstALL):
            # Convergence
            if gen_stats.optimal_loose_count > 0 and gen_stats.optimal_count == 0:
                self.NI_loose = gen_i
                self.optSaved_NI_loose = gen_stats.optimal_loose_count
            self.Num_loose += gen_stats.optimal_loose_count

            # Selection Intensity
            if gen_i == 0:
                self.I_start = gen_stats.intensity
            if self.I_min is None or gen_stats.intensity < self.I_min:
                self.I_min = gen_stats.intensity
                self.NI_I_min = gen_i
            if self.I_max is None or gen_stats.intensity > self.I_max:
                self.I_max = gen_stats.intensity
                self.NI_I_max = gen_i
            if self.I_avg is None:
                self.I_avg = gen_stats.intensity
            else:
                self.I_avg = (self.I_avg * (gen_i - 1) + gen_stats.intensity) / gen_i

            # Selection Difference
            if gen_i == 0:
                self.s_start = gen_stats.difference
            if self.s_min is None or gen_stats.difference < self.s_min:
                self.s_min = gen_stats.difference
                self.NI_s_min = gen_i
            if self.s_max is None or gen_stats.difference > self.s_max:
                self.s_max = gen_stats.difference
                self.NI_s_max = gen_i
            if self.s_avg is None:
                self.s_avg = gen_stats.difference
            else:
                self.s_avg = (self.s_avg * (gen_i - 1) + gen_stats.difference) / gen_i

            # Growth Rate
            if gen_i == 0:
                self.GR_start = gen_stats.growth_rate
            if gen_i == 2:
                self.GR_early = gen_stats.growth_rate
            if self.GR_late is None and gen_stats.num_of_best >= N / 2:
                self.GR_late = gen_stats.growth_rate
                self.NI_GR_late = gen_i
            if self.GR_avg is None:
                self.GR_avg = gen_stats.growth_rate
            else:
                self.GR_avg = (self.GR_avg * (gen_i - 1) + gen_stats.growth_rate) / gen_i

            if gen_i == 0:
                self.Pr_start = gen_stats.Pr
            if self.Pr_min is None or gen_stats.Pr < self.Pr_min:
                self.Pr_min = gen_stats.Pr
                self.NI_Pr_min = gen_i
            if self.Pr_max is None or gen_stats.Pr > self.Pr_max:
                self.Pr_max = gen_stats.Pr
                self.NI_Pr_max = gen_i
            if self.Pr_avg is None:
                self.Pr_avg = gen_stats.Pr
            else:
                self.Pr_avg = (self.Pr_avg * (gen_i - 1) + gen_stats.Pr) / gen_i

            if gen_i == 0:
                self.Fish_start = gen_stats.Fish
            if self.Fish_min is None or gen_stats.Fish < self.Fish_min:
                self.Fish_min = gen_stats.Fish
                self.NI_Fish_min = gen_i
            if self.Fish_max is None or gen_stats.Fish > self.Fish_max:
                self.Fish_max = gen_stats.Fish
                self.NI_Fish_max = gen_i
            if self.Fish_avg is None:
                self.Fish_avg = gen_stats.Fish
            else:
                self.Fish_avg = (self.Fish_avg * (gen_i - 1) + gen_stats.Fish) / gen_i

            if gen_i == 0:
                self.Kend_start = gen_stats.Kend
            if self.Kend_min is None or gen_stats.Kend < self.Kend_min:
                self.Kend_min = gen_stats.Kend
                self.NI_Kend_min = gen_i
            if self.Kend_max is None or gen_stats.Kend > self.Kend_max:
                self.Kend_max = gen_stats.Kend
                self.NI_Kend_max = gen_i
            if self.Kend_avg is None:
                self.Kend_avg = gen_stats.Kend
            else:
                self.Kend_avg = (self.Kend_avg * (gen_i - 1) + gen_stats.Kend) / gen_i
            if math.isnan(self.Kend_avg):
                pass


    def update_final_stats(self, gen_stats: GenerationStats, gen_i):

        if not isinstance(self.experiment_params.fitness_function, FconstALL):
            self.F_found = gen_stats.f_best
            self.F_avg = gen_stats.f_avg

            if gen_i == 2:
                self.GR_early = gen_stats.growth_rate
            if self.GR_late is None and gen_stats.num_of_best >= N / 2:
                self.GR_late = gen_stats.growth_rate
                self.NI_GR_late = gen_i
            if self.GR_avg is None:
                self.GR_avg = gen_stats.growth_rate
            else:
                self.GR_avg = (self.GR_avg * (gen_i - 1) + gen_stats.growth_rate) / gen_i
