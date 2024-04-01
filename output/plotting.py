import matplotlib.pyplot as plt
from config import N, OUTPUT_FOLDER
import os

from model.experiment_helpers import ExperimentParams
from model.population import Population
from stats.generation_stats import GenerationStats
import numpy as np


def plot_run_stats(
        gen_stats_list: list[GenerationStats],
        experiment_params: ExperimentParams,
        run_i):
    reproduction_rates = [gen_stats.reproduction_rate for gen_stats in gen_stats_list if gen_stats.reproduction_rate is not None]
    losses_of_diversity = [gen_stats.loss_of_diversity for gen_stats in gen_stats_list if gen_stats.loss_of_diversity is not None]
    __plot_stat2(reproduction_rates, losses_of_diversity, experiment_params, run_i, 'Reproduction Rate', 'Loss of Diversity', 'rr_and_lod')

    if experiment_params.fitness_function.name() != 'FconstALL':
        f_avgs = [gen_stats.f_avg for gen_stats in gen_stats_list]
        __plot_stat(f_avgs, experiment_params, run_i, 'Fitness Average', 'f_avg')

        f_bests = [gen_stats.f_best for gen_stats in gen_stats_list]
        __plot_stat(f_bests, experiment_params, run_i, 'Highest Fitness', 'f_best')

        intensities = [gen_stats.intensity for gen_stats in gen_stats_list if gen_stats.intensity is not None]
        __plot_stat(intensities, experiment_params, run_i, 'Selection Intensity', 'intensity')

        differences = [gen_stats.difference for gen_stats in gen_stats_list if gen_stats.difference is not None]
        __plot_stat(differences, experiment_params, run_i, 'Selection Difference', 'difference')

        __plot_stat2(differences, intensities, experiment_params, run_i, 'Difference', 'Intensity', 'intensity_and_difference')

        f_stds = [gen_stats.f_std for gen_stats in gen_stats_list]
        __plot_stat(f_stds, experiment_params, run_i, 'Fitness Standard Deviation', 'f_std')

        optimal_counts = [gen_stats.optimal_count for gen_stats in gen_stats_list]
        __plot_stat(optimal_counts, experiment_params, run_i, 'Number of Optimal Chromosomes', 'optimal_count')

        growth_rates = [gen_stats.growth_rate for gen_stats in gen_stats_list]
        if len(growth_rates) > 0:
            growth_rates = growth_rates[1:]
        __plot_stat(growth_rates, experiment_params, run_i, 'Growth Rate', 'growth_rate')

        __plot_stat(
            [gen_stats.Pr for gen_stats in gen_stats_list],
            experiment_params, run_i, 'Number of unique chromosomes', 'Pr')

        __plot_stat(
            [gen_stats.num_of_best for gen_stats in gen_stats_list],
            experiment_params, run_i, 'Num of best individuals', 'num_of_best')

        __plot_stat(
            [gen_stats.Fish for gen_stats in gen_stats_list],
            experiment_params, run_i, 'Fisher exact test', 'fisher')

        __plot_stat(
            [gen_stats.Kend for gen_stats in gen_stats_list],
            experiment_params, run_i, 'Kendall’s τ-b', 'kendall')

    __plot_stat(
        [gen_stats.number_of_unique_chromosomes for gen_stats in gen_stats_list],
        experiment_params, run_i, 'Number of unique chromosomes', 'unique_X')


def plot_generation_stats(
        population: Population,
        experiment_params: ExperimentParams,
        run_i, gen_i):
    __plot_genotype_distribution(population, experiment_params, run_i, gen_i)
    if experiment_params.fitness_function.name() != 'FconstALL':
        __plot_fitness_distribution(population, experiment_params, run_i, gen_i)
    if experiment_params.fitness_function.name() not in ['FconstALL', 'FH']:
        __plot_phenotype_distribution(population, experiment_params, run_i, gen_i)


def __plot_stat(
        data,
        experiment_params: ExperimentParams,
        run_i,
        ylabel,
        file_name):
    param_hierarchy = __get_path_hierarchy(experiment_params, run_i)
    path = '/'.join(param_hierarchy)

    if not os.path.exists(path):
        os.makedirs(path)

    plt.plot(data)
    plt.ylabel(ylabel)
    plt.xlabel('Generation')
    plt.savefig(f'{path}/{file_name}.png')
    plt.close()


def __plot_stat2(
        data1, data2,
        experiment_params: ExperimentParams,
        run_i,
        label1, label2,
        file_name):
    param_hierarchy = __get_path_hierarchy(experiment_params, run_i)
    path = '/'.join(param_hierarchy)

    if not os.path.exists(path):
        os.makedirs(path)

    plt.plot(data1, label=label1)
    plt.plot(data2, label=label2)

    plt.xlabel('Generation')
    plt.legend()
    plt.savefig(f'{path}/{file_name}.png')
    plt.close()


def __plot_fitness_distribution(
        population: Population,
        experiment_params: ExperimentParams,
        run_i, gen_i):
    param_hierarchy = __get_path_hierarchy(experiment_params, run_i) + ['fitness']
    path = '/'.join(param_hierarchy)

    if not os.path.exists(path):
        os.makedirs(path)

    x_max = population.fitness_function.get_optimal().fitness
    x_step = x_max / 100
    (x, y) = __get_distribution(population.fitnesses, x_max=x_max, x_step=x_step)
    plt.bar(x, y, width=x_step*0.8)
    plt.xlabel('Chromosome fitness')
    plt.ylabel('Number of chromosomes')
    plt.savefig(f'{path}/{gen_i}.png')
    plt.close()


def __plot_phenotype_distribution(
        population: Population,
        experiment_params: ExperimentParams,
        run_i, gen_i):
    param_hierarchy = __get_path_hierarchy(experiment_params, run_i) + ['phenotype']
    path = '/'.join(param_hierarchy)

    if not os.path.exists(path):
        os.makedirs(path)

    phenotypes = [population.fitness_function.get_phenotype(geno) for geno in population.genotypes]
    encoder = population.fitness_function.encoder
    x_min = encoder.lower_bound
    x_max = encoder.upper_bound
    x_step = (x_max - x_min) / 100
    (x, y) = __get_distribution(phenotypes, x_min=x_min, x_max=x_max, x_step=x_step)
    plt.bar(x, y, width=x_step*0.8)
    plt.xlabel('Chromosome phenotype')
    plt.ylabel('Number of chromosomes')
    plt.savefig(f'{path}/{gen_i}.png')
    plt.close()


def __plot_genotype_distribution(
        population: Population,
        experiment_params: ExperimentParams,
        run_i, gen_i):
    param_hierarchy = __get_path_hierarchy(experiment_params, run_i) + ['genotype']
    path = '/'.join(param_hierarchy)

    if not os.path.exists(path):
        os.makedirs(path)

    ones_counts = [len([True for gene in geno if gene == b'1']) for geno in population.genotypes]
    (x, y) = __get_distribution(ones_counts, x_max=population.fitness_function.chr_length)
    plt.bar(x, y)
    plt.xlabel('Number of 1s in genotype')
    plt.ylabel('Number of chromosomes')
    plt.savefig(f'{path}/{gen_i}.png')
    plt.close()


def __get_distribution(data, x_min=0, x_max=None, x_step=1):
    if x_max is None:
        x_max = max(data)

    x = np.arange(x_min, x_max + x_step, x_step)
    y = np.zeros_like(x)
    for val in data:
        idx = int(round((val - x_min) / x_step))
        idx = max(0, min(idx, len(x)-1))
        y[idx] += 1

    return (x, y)


def __get_path_hierarchy(experiment_params: ExperimentParams, run_i):
    return [
        OUTPUT_FOLDER,
        'graphs',
        experiment_params.fitness_function.name(),
        str(N),
        experiment_params.selection_method.name(),
        experiment_params.genetic_operator.name(),
        str(experiment_params.population_initialization),
        str(run_i)
    ]
