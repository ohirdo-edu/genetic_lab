from itertools import starmap

from config import env, NR, get_pop_seed
from model.experiment_helpers import ExperimentParams
from model.fitness_functions import *
from selection.selection_method import RWS, SUS, TruncatedRWS, TruncatedSUS
from model.encoding import *
from model.gen_operators import *
from output import excel
from runner import run_experiment
from datetime import datetime
from multiprocessing import Pool
import time

from model.population_initialization import ConstantPopulationInitialization, PercentagePopulationInitialization

if env == 'test':
    fitness_functions = [
        # FconstALL(100),
        # FH(100),
        FDebb4(FloatEncoder(0, 1.023, 10)),
    ]
    selection_methods = [
        # RWS(),
        SUS(),
        # TruncatedRWS(T=0.3),
        # TruncatedSUS(T=0.3),
    ]
    gen_operators = [
        BlankGenOperator(),
    ]

    def gen_initialization_methods(fitness_function: FitnessFunc, genetic_operator: GeneticOperator):
        # if isinstance(fitness_function, FconstALL):
        #     yield ConstantPopulationInitialization(constant_optimal_size=0)
        #     return
        #
        # if not isinstance(genetic_operator, BlankGenOperator):
        #     yield ConstantPopulationInitialization(constant_optimal_size=0)

        yield ConstantPopulationInitialization(constant_optimal_size=1)
        # yield PercentagePopulationInitialization(optimal_size_percent=10)
else:
    fitness_functions = [
        FconstALL(100),
        FH(100),

        Fx2(FloatEncoder(0.0, 10.23, 10)),
        Fx2(FloatEncoder(0.0, 10.23, 10, is_gray=True)),
        F5122subx2(FloatEncoder(-5.12, 5.11, 10)),
        F5122subx2(FloatEncoder(-5.12, 5.11, 10, is_gray=True)),
        Fexp(0.25, FloatEncoder(0.0, 10.23, 10)),
        Fexp(0.25, FloatEncoder(0.0, 10.23, 10, is_gray=True)),
        Fexp(1, FloatEncoder(0.0, 10.23, 10)),
        Fexp(1, FloatEncoder(0.0, 10.23, 10, is_gray=True)),
        Fexp(2, FloatEncoder(0.0, 10.23, 10)),
        Fexp(2, FloatEncoder(0.0, 10.23, 10, is_gray=True)),

        FRastrigin(FloatEncoder(-5.12, 5.11, 10)),
        FRastrigin(FloatEncoder(-5.12, 5.11, 10, is_gray=True)),
        FDebb2(FloatEncoder(0, 1.023, 10)),
        FDebb2(FloatEncoder(0, 1.023, 10, is_gray=True)),
        FDebb4(FloatEncoder(0, 1.023, 10)),
        FDebb4(FloatEncoder(0, 1.023, 10, is_gray=True)),
    ]

    selection_methods = [
        RWS(),
        SUS(),

        TruncatedRWS(T=0.2),
        TruncatedRWS(T=0.4),
        TruncatedRWS(T=0.5),

        TruncatedSUS(T=0.2),
        TruncatedSUS(T=0.4),
        TruncatedSUS(T=0.5),
    ]
    gen_operators = [
        BlankGenOperator(),
    ]

    def gen_initialization_methods(fitness_function: FitnessFunc, genetic_operator: GeneticOperator):
        if isinstance(fitness_function, FconstALL):
            yield ConstantPopulationInitialization(constant_optimal_size=0)
            return

        if not isinstance(genetic_operator, BlankGenOperator):
            yield ConstantPopulationInitialization(constant_optimal_size=0)

        yield ConstantPopulationInitialization(constant_optimal_size=1)
        yield ConstantPopulationInitialization(constant_optimal_size=5)
        yield PercentagePopulationInitialization(optimal_size_percent=10)

# a list of tuples of parameters for each run that involves a certain fitness function 
# {fitness_func_name: [(tuples with run parameters), (), ..., ()], other_func: [], ...}
experiment_params_by_fitness_func = {
    fitness_function: [
        ExperimentParams(
            fitness_function=fitness_function,
            genetic_operator=gen_operator,
            selection_method=selection_method,
            population_initialization=init_method,
        )
        for selection_method in selection_methods
        for gen_operator in gen_operators
        for init_method in gen_initialization_methods(fitness_function, gen_operator)
    ] for fitness_function in fitness_functions
}


def log(x):
    datetime_prefix = str(datetime.now())[:-4]
    print(f'{datetime_prefix} | {x}')


def main():
    log('Program start')
    print('----------------------------------------------------------------------')
    start_time = time.time()
    results = []

    # starmap_func = starmap
    p = Pool(processes=2)
    starmap_func = p.starmap

    for ff, experiment_params_list in experiment_params_by_fitness_func.items():
        ff_start_time = time.time()
        populations_cache: dict[str, [Population]] = {}
        params_and_populations: list[tuple[ExperimentParams, [Population]]] = []
        for experiment_params in experiment_params_list:
            key = str(experiment_params.population_initialization)
            populations = populations_cache.get(key)
            if populations is None:
                populations = [Population(ff, seed=get_pop_seed(run_i), initialization=experiment_params.population_initialization)
                               for run_i in range(NR)]
                populations_cache[key] = populations

            params_and_populations.append((experiment_params, populations))

        experiment_stats_list = list(starmap_func(run_experiment, params_and_populations))

        # experiment_stats_list = [run_experiment(experiment_params, populations)
        #                          for experiment_params, populations in params_and_populations]

        excel.write_ff_stats(experiment_stats_list)
        for experiment_stats in experiment_stats_list:
            del experiment_stats.runs
            results.append(experiment_stats)

        ff_end_time = time.time()
        ff_name = ff.name()
        log(f'{ff_name} experiments finished in {(ff_end_time - ff_start_time):.2f}s')

    excel.write_aggregated_stats(results)

    print('----------------------------------------------------------------------')
    end_time = time.time()
    log(f'Program end. Total runtime: {end_time - start_time:.2f}s')


if __name__ == '__main__':
    main()
