from config import NR
from model.experiment_helpers import ExperimentParams
from stats.run_stats import RunStats
import numpy as np


class ExperimentStats:
    def __init__(self, experiment_params: ExperimentParams):
        self.experiment_params = experiment_params
        self.runs = np.empty(NR, dtype=object)

        self.Suc = 0
        self.N_Suc = 0
        self.Min_NI = None
        self.Max_NI = None
        self.Avg_NI = None
        self.Sigma_NI = None

        self.Avg_NI_loose = None
        self.Sigma_NI_loose = None
        self.Avg_optSaved_NI_loose = None
        self.Sigma_optSaved_NI_loose = None
        self.Avg_Num_loose = None
        self.Sigma_Num_loose = None

        self.nonSuc = None
        self.nonMin_NI = None
        self.nonMax_NI = None
        self.nonAvg_NI = None
        self.nonSigma_NI = None
        self.nonAvg_F_found = None
        self.nonSigma_F_found = None
        self.nonMax_F_found = None
        self.nonAvg_NI_loose = None
        self.nonSigma_NI_loose = None
        self.nonAvg_optSaved_NI_loose = None
        self.nonSigma_optSaved_NI_loose = None
        self.nonAvg_Num_loose = None
        self.nonSigma_Num_loose = None

        # Reproduction Rate
        self.Min_RR_min = None
        self.NI_RR_min = None
        self.Max_RR_max = None
        self.NI_RR_max = None
        self.Avg_RR_min = None
        self.Avg_RR_max = None
        self.Avg_RR_avg = None
        self.Sigma_RR_min = None
        self.Sigma_RR_max = None
        self.Sigma_RR_avg = None

        # Loss of Diversity
        self.Min_Teta_min = None
        self.NI_Teta_min = None
        self.Max_Teta_max = None
        self.NI_Teta_max = None
        self.Avg_Teta_min = None
        self.Avg_Teta_max = None
        self.Avg_Teta_avg = None
        self.Sigma_Teta_min = None
        self.Sigma_Teta_max = None
        self.Sigma_Teta_avg = None

        # Selection Intensity
        self.Min_I_min = None
        self.NI_I_min = None
        self.Max_I_max = None
        self.NI_I_max = None
        self.Avg_I_min = None
        self.Avg_I_max = None
        self.Avg_I_avg = None
        self.Sigma_I_min = None
        self.Sigma_I_max = None
        self.Sigma_I_avg = None

        self.Min_I_start = None
        self.Max_I_start = None
        self.Avg_I_start = None
        self.Sigma_I_start = None

        # Selection Difference
        self.Min_s_min = None
        self.NI_s_min = None
        self.Max_s_max = None
        self.NI_s_max = None
        self.Avg_s_min = None
        self.Avg_s_max = None
        self.Avg_s_avg = None
        self.Min_s_start = None
        self.Max_s_start = None
        self.Avg_s_start = None
        self.Sigma_s_start = None

        # Growth Rate
        self.Min_GR_early = None
        self.Max_GR_early = None
        self.Avg_GR_early = None
        self.Min_GR_late = None
        self.Max_GR_late = None
        self.Avg_GR_late = None
        self.Min_GR_avg = None
        self.Max_GR_avg = None
        self.Avg_GR_avg = None

        self.Min_GR_start = None
        self.Max_GR_start = None
        self.Avg_GR_start = None
        self.Sigma_GR_start = None

        self.Min_Pr_min = None
        self.NI_Pr_min = None
        self.Max_Pr_max = None
        self.NI_Pr_max = None
        self.Avg_Pr_min = None
        self.Avg_Pr_max = None
        self.Avg_Pr_avg = None
        self.Sigma_Pr_max = None
        self.Sigma_Pr_min = None
        self.Sigma_Pr_avg = None
        self.Min_Pr_start = None
        self.Max_Pr_start = None
        self.Avg_Pr_start = None
        self.Sigma_Pr_start = None

        self.Min_Fish_min = None
        self.NI_Fish_min = None
        self.Max_Fish_max = None
        self.NI_Fish_max = None
        self.Avg_Fish_min = None
        self.Avg_Fish_max = None
        self.Avg_Fish_avg = None
        self.Sigma_Fish_max = None
        self.Sigma_Fish_min = None
        self.Sigma_Fish_avg = None
        self.Min_Fish_start = None
        self.Max_Fish_start = None
        self.Avg_Fish_start = None
        self.Sigma_Fish_start = None

        self.Min_Kend_min = None
        self.NI_Kend_min = None
        self.Max_Kend_max = None
        self.NI_Kend_max = None
        self.Avg_Kend_min = None
        self.Avg_Kend_max = None
        self.Avg_Kend_avg = None
        self.Sigma_Kend_max = None
        self.Sigma_Kend_min = None
        self.Sigma_Kend_avg = None
        self.Min_Kend_start = None
        self.Max_Kend_start = None
        self.Avg_Kend_start = None
        self.Sigma_Kend_start = None

        self.Avg_RR_start = None
        self.Avg_RR_fin = None
        self.Sigma_RR_start = None
        self.Sigma_RR_fin = None
        self.Min_RR_start = None
        self.Max_RR_start = None

        self.Avg_Teta_start = None
        self.Avg_Teta_fin = None
        self.Sigma_Teta_start = None
        self.Sigma_Teta_fin = None
        self.Min_Teta_start = None
        self.Max_Teta_start = None

        self.Avg_unique_X_start = None
        self.Avg_unique_X_fin = None
        self.Sigma_unique_X_start = None
        self.Sigma_unique_X_fin = None
        self.Min_unique_X_start = None
        self.Max_unique_X_start = None
        self.Min_unique_X_fin = None
        self.Max_unique_X_fin = None


    def add_run(self, run: RunStats, run_i):
        self.runs[run_i] = run

    def calculate(self):
        successful_runs = [run for run in self.runs if run.is_successful]
        unsuccessful_but_converged_runs = [run for run in self.runs if not run.is_successful and run.has_converged]
        self.N_Suc = len(successful_runs)
        self.Suc = self.N_Suc / NR
        self.nonSuc = len(unsuccessful_but_converged_runs) / NR

        self.__calculate_convergence_stats(successful_runs, unsuccessful_but_converged_runs)
        self.__calculate_rr_stats(successful_runs)
        self.__calculate_teta_stats(successful_runs)

        if self.experiment_params.should_calculate_pressure_stats():
            self.__calculate_s_stats(successful_runs)
            self.__calculate_i_stats(successful_runs)
            self.__calculate_gr_stats(successful_runs)
            self.__calculate_new_pr_stats_for(successful_runs, 'Pr')
            self.__calculate_new_pr_stats_for(successful_runs, 'Fish')
            self.__calculate_new_pr_stats_for(successful_runs, 'Kend')


    def __calculate_convergence_stats(self, successful_runs: list[RunStats], unsuccessful_but_converged_runs: list[RunStats]):
        NIs = [run.NI for run in successful_runs]
        if NIs:
            self.Min_NI = min(NIs)
            self.Max_NI = max(NIs)
            self.Avg_NI = np.mean(NIs)
            self.Sigma_NI = np.std(NIs)

        if self.experiment_params.should_calculate_convergence_stats():
            successful_runs_with_optimal_looses = [run for run in successful_runs if run.Num_loose > 0]
            if successful_runs_with_optimal_looses:
                self.Avg_NI_loose = np.mean([run.NI_loose for run in successful_runs_with_optimal_looses])
                self.Sigma_NI_loose = np.std([run.NI_loose for run in successful_runs_with_optimal_looses])
                self.Avg_optSaved_NI_loose = np.mean([run.optSaved_NI_loose for run in successful_runs_with_optimal_looses])
                self.Sigma_optSaved_NI_loose = np.std([run.optSaved_NI_loose for run in successful_runs_with_optimal_looses])
                self.Avg_Num_loose = np.mean([run.Num_loose for run in successful_runs_with_optimal_looses])
                self.Sigma_Num_loosee = np.std([run.Num_loose for run in successful_runs_with_optimal_looses])

            if unsuccessful_but_converged_runs:
                self.nonMin_NI = min([run.NI for run in unsuccessful_but_converged_runs])
                self.nonMax_NI = max([run.NI for run in unsuccessful_but_converged_runs])
                self.nonAvg_NI = np.mean([run.NI for run in unsuccessful_but_converged_runs])
                self.nonSigma_NI = np.std([run.NI for run in unsuccessful_but_converged_runs])

                self.nonAvg_F_found = np.mean([run.F_found for run in unsuccessful_but_converged_runs])
                self.nonSigma_F_found = np.std([run.F_found for run in unsuccessful_but_converged_runs])
                self.nonMax_F_found = max([run.F_found for run in unsuccessful_but_converged_runs])

                runs_with_optimal_looses = [run for run in unsuccessful_but_converged_runs if run.Num_loose > 0]
                if runs_with_optimal_looses:
                    self.nonAvg_NI_loose = np.mean([run.NI_loose for run in runs_with_optimal_looses])
                    self.nonSigma_NI_loose = np.std([run.NI_loose for run in runs_with_optimal_looses])
                    self.nonAvg_optSaved_NI_loose = np.mean([run.optSaved_NI_loose for run in runs_with_optimal_looses])
                    self.nonSigma_optSaved_NI_loose = np.std([run.optSaved_NI_loose for run in runs_with_optimal_looses])
                    self.nonAvg_Num_loose = np.mean([run.Num_loose for run in runs_with_optimal_looses])
                    self.nonSigma_Num_loose = np.std([run.Num_loose for run in runs_with_optimal_looses])


    def __calculate_rr_stats(self, runs: list[RunStats]):
        RR_min_list = [run.RR_min for run in runs]
        if RR_min_list:
            run_i_RR_min = np.argmin(RR_min_list)
            self.NI_RR_min = runs[run_i_RR_min].NI_RR_min
            self.Min_RR_min = RR_min_list[run_i_RR_min]
            self.Avg_RR_min = np.mean(RR_min_list)
            self.Sigma_RR_min = np.std(RR_min_list)
        RR_max_list = [run.RR_max for run in runs]
        if RR_max_list:
            run_i_RR_max = np.argmax(RR_max_list)
            self.NI_RR_max = runs[run_i_RR_max].NI_RR_max
            self.Max_RR_max = RR_max_list[run_i_RR_max]
            self.Avg_RR_max = np.mean(RR_max_list)
            self.Sigma_RR_max = np.std(RR_max_list)
        RR_avg_list = [run.RR_avg for run in runs]
        if RR_avg_list:
            self.Avg_RR_avg = np.mean(RR_avg_list)
            self.Sigma_RR_avg = np.std(RR_avg_list)

    def __calculate_teta_stats(self, runs: list[RunStats]):
        Teta_min_list = [run.Teta_min for run in runs]
        if Teta_min_list:
            run_i_Teta_min = np.argmin(Teta_min_list)
            self.NI_Teta_min = runs[run_i_Teta_min].NI_Teta_min
            self.Min_Teta_min = Teta_min_list[run_i_Teta_min]
            self.Avg_Teta_min = np.mean(Teta_min_list)
            self.Sigma_Teta_min = np.std(Teta_min_list)
        Teta_max_list = [run.Teta_max for run in runs]
        if Teta_max_list:
            run_i_Teta_max = np.argmax(Teta_max_list)
            self.NI_Teta_max = runs[run_i_Teta_max].NI_Teta_max
            self.Max_Teta_max = Teta_max_list[run_i_Teta_max]
            self.Avg_Teta_max = np.mean(Teta_max_list)
            self.Sigma_Teta_max = np.std(Teta_max_list)
        Teta_avg_list = [run.Teta_avg for run in runs]
        if Teta_avg_list:
            self.Avg_Teta_avg = np.mean(Teta_avg_list)
            self.Sigma_Teta_avg = np.std(Teta_avg_list)

        for attr in ['RR', 'Teta']:
            for suffix in ['fin', 'start']:
                self.__calculate_stat(runs, f'{attr}_{suffix}', 'Sigma', np.std)
                self.__calculate_stat(runs, f'{attr}_{suffix}', 'Avg', np.mean)
            self.__calculate_stat(runs, f'{attr}_start', 'Min', min)
            self.__calculate_stat(runs, f'{attr}_start', 'Max', max)

        for suffix in ['fin', 'start']:
            attr = 'unique_X'
            self.__calculate_stat(runs, f'{attr}_{suffix}', 'Sigma', np.std)
            self.__calculate_stat(runs, f'{attr}_{suffix}', 'Avg', np.mean)
            self.__calculate_stat(runs, f'{attr}_{suffix}', 'Min', min)
            self.__calculate_stat(runs, f'{attr}_{suffix}', 'Max', max)

    def __calculate_s_stats(self, runs: list[RunStats]):
        s_start_list = [run.s_start for run in runs]
        if s_start_list:
            self.Min_s_start = min(s_start_list)
            self.Max_s_start = max(s_start_list)
            self.Avg_s_start = np.mean(s_start_list)
            self.Sigma_s_start = np.std(s_start_list)

        s_min_list = [run.s_min for run in runs]
        if s_min_list:
            run_i_s_min = np.argmin(s_min_list)
            self.NI_s_min = runs[run_i_s_min].NI_s_min
            self.Min_s_min = s_min_list[run_i_s_min]
            self.Avg_s_min = np.mean(s_min_list)
        s_max_list = [run.s_max for run in runs]
        if s_max_list:
            run_i_s_max = np.argmax(s_max_list)
            self.NI_s_max = runs[run_i_s_max].NI_s_max
            self.Max_s_max = s_max_list[run_i_s_max]
            self.Avg_s_max = np.mean(s_max_list)
        s_avg_list = [run.s_avg for run in runs]
        if s_avg_list:
            self.Avg_s_avg = np.mean(s_avg_list)

    def __calculate_i_stats(self, runs: list[RunStats]):
        I_min_list = [run.I_min for run in runs]
        if I_min_list:
            run_i_I_min = np.argmin(I_min_list)
            self.NI_I_min = runs[run_i_I_min].NI_I_min
            self.Min_I_min = I_min_list[run_i_I_min]
            self.Avg_I_min = np.mean(I_min_list)
            self.Sigma_I_min = np.std(I_min_list)
        I_max_list = [run.I_max for run in runs]
        if I_max_list:
            run_i_I_max = np.argmax(I_max_list)
            self.NI_I_max = runs[run_i_I_max].NI_I_max
            self.Max_I_max = I_max_list[run_i_I_max]
            self.Avg_I_max = np.mean(I_max_list)
            self.Sigma_I_max = np.std(I_max_list)
        I_avg_list = [run.I_avg for run in runs]
        if I_avg_list:
            self.Avg_I_avg = np.mean(I_avg_list)
            self.Sigma_I_avg = np.std(I_avg_list)

        I_start_list =  [run.I_start for run in runs]
        if I_start_list:
            self.Min_I_start = min(I_start_list)
            self.Max_I_start = max(I_start_list)
            self.Avg_I_start = np.mean(I_start_list)
            self.Sigma_I_start = np.std(I_start_list)

    def __calculate_gr_stats(self, runs: list[RunStats]):
        gre_list = [run.GR_early for run in runs]
        grl_list = [run.GR_late for run in runs if run.GR_late is not None]
        gra_list = [run.GR_avg for run in runs]
        if gre_list:
            self.Avg_GR_early = np.mean(gre_list)
            self.Min_GR_early = min(gre_list)
            self.Max_GR_early = max(gre_list)
        if grl_list:
            self.Avg_GR_late = np.mean(grl_list)
            self.Min_GR_late = min(grl_list)
            self.Max_GR_late = max(grl_list)
        if gra_list:
            self.Avg_GR_avg = np.mean(gra_list)
            self.Min_GR_avg = min(gra_list)
            self.Max_GR_avg = max(gra_list)

        GR_start_list = [run.GR_start for run in runs]
        if GR_start_list:
            self.Min_GR_start = min(GR_start_list)
            self.Max_GR_start = max(GR_start_list)
            self.Avg_GR_start = np.mean(GR_start_list)
            self.Sigma_GR_start = np.std(GR_start_list)

    def __calculate_new_pr_stats_for(self, runs: list[RunStats], attr_name: str):
        min_list = [getattr(run, f"{attr_name}_min") for run in runs]
        if min_list:
            run_i_min = np.argmin(min_list)
            setattr(self, f"NI_{attr_name}_min", getattr(runs[run_i_min], f"NI_{attr_name}_min"))
            setattr(self, f"Min_{attr_name}_min", min_list[run_i_min])
            setattr(self, f"Avg_{attr_name}_min", np.mean(min_list))
            setattr(self, f"Sigma_{attr_name}_min", np.std(min_list))

        max_list = [getattr(run, f"{attr_name}_max") for run in runs]
        if max_list:
            run_i_max = np.argmax(max_list)
            setattr(self, f"NI_{attr_name}_max", getattr(runs[run_i_max], f"NI_{attr_name}_max"))
            setattr(self, f"Max_{attr_name}_max", max_list[run_i_max])
            setattr(self, f"Avg_{attr_name}_max", np.mean(max_list))
            setattr(self, f"Sigma_{attr_name}_max", np.std(max_list))

        avg_list = [getattr(run, f"{attr_name}_avg") for run in runs]
        if avg_list:
            setattr(self, f"Avg_{attr_name}_avg", np.mean(avg_list))
            setattr(self, f"Sigma_{attr_name}_avg", np.std(avg_list))

        start_list = [getattr(run, f"{attr_name}_start") for run in runs]
        if start_list:
            setattr(self, f"Min_{attr_name}_start", min(start_list))
            setattr(self, f"Max_{attr_name}_start", max(start_list))
            setattr(self, f"Avg_{attr_name}_start", np.mean(start_list))
            setattr(self, f"Sigma_{attr_name}_start", np.std(start_list))

    def __calculate_stat(self, runs: list[RunStats], attr_name: str, prefix: str, aggregate):
        attr_values = [getattr(run, f"{attr_name}") for run in runs]
        if attr_values:
            setattr(self, f"{prefix}_{attr_name}", aggregate(attr_values))

    def __str__(self):
        return ("Suc: " + str(self.Suc) + "%" +
                "\nMin: " + str(self.Min_NI) + "\nMax: " + str(self.Max_NI) + "\nAvg: " + str(self.Avg_NI))
