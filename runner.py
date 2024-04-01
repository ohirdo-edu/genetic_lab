from multiprocessing import Pool
import time
from itertools import starmap
import gc

from config import NR, THREADS
from stats.experiment_stats import ExperimentStats
from evo_algorithm import EvoAlgorithm
from model.population import Population
from copy import deepcopy
from datetime import datetime
from model.experiment_helpers import ExperimentParams


def run_experiment(experiment_params: ExperimentParams,
                   populations: list[Population]):
    start_time = time.time()
    stats = ExperimentStats(experiment_params)

    run_param_list = [
        (populations[run_i],
         experiment_params,
         run_i
        )
        for run_i in range(NR)
    ]

    results = starmap(run, run_param_list)
    for run_i, run_stats in results:
        stats.add_run(run_stats, run_i)

    # for i in range(NR // THREADS):
    #     with Pool(THREADS) as p:
    #         results = p.starmap(run, run_param_list[(i * THREADS):((i+1) * THREADS)])
    #         for run_i, run_stats in results:
    #             stats.add_run(run_stats, run_i)
    # if NR % THREADS:
    #     with Pool(NR % THREADS) as p:
    #         results = p.starmap(run, run_param_list[-(NR % THREADS):])
    #         for run_i, run_stats in results:
    #             stats.add_run(run_stats, run_i)
    
    stats.calculate()
    finish_time = time.time()
    print(f'{str(datetime.now())[:-4]} | Experiment ({experiment_params}) finished in {(finish_time - start_time):.2f}s')
    gc.collect()
    return stats


def run(init_population: Population,
        experiment_params: ExperimentParams,
        run_i: int):
    current_run = EvoAlgorithm(deepcopy(init_population), experiment_params).run(run_i)
    return run_i, current_run
