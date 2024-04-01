from functools import cache

import numpy as np
from config import DELTA, SIGMA
from model.chromosome import Chromosome
from model.encoding import Encoder
from math import exp
import math


class FitnessFunc:
    def __init__(self, chr_length):
        self.chr_length = chr_length
        self.optimal = None

    def apply(self, genotype):
        raise NotImplementedError()

    def get_optimal(self):
        raise NotImplementedError()
    
    def get_phenotype(self, genotype):
        raise NotImplementedError()
    
    def check_chromosome_success(self, chr: Chromosome):
        y_diff = abs(chr.fitness - self.get_optimal().fitness)
        x_diff = abs(self.get_phenotype(chr.genotype) - self.get_phenotype(self.get_optimal().genotype))
        return y_diff <= DELTA and x_diff <= SIGMA

    def name(self) -> str:
        return self.__class__.__name__


class FconstALL(FitnessFunc):
    def apply(self, genotype):
        return 100

    def get_optimal(self):
        if not self.optimal:
            self.optimal = Chromosome(0, np.full(self.chr_length, b'0'), self)
        return self.optimal
    
    def get_phenotype(self, genotype):
        return 0
    
    def check_chromosome_success(self, ch):
        return True

    def name(self) -> str:
        return self.__class__.__name__


class FH(FitnessFunc):
    def apply(self, genotype):
        k = len([True for gene in genotype if gene == b'1'])
        return self.chr_length - k

    def get_optimal(self):
        if not self.optimal:
            self.optimal = Chromosome(0, np.full(self.chr_length, b'0'), self)
        return self.optimal

    def get_phenotype(self, genotype):
        return len([True for gene in genotype if gene == b'1'])


class Freal(FitnessFunc):
    def __init__(self, encoder: Encoder):
        super().__init__(encoder.length)
        self.encoder = encoder
        self.is_caching = encoder.length <= 12
        self.cache_dict = {}
        if self.is_caching:
            for v in self.encoder.get_all_values():
                self.cache_dict[v.tobytes()] = self.calculate(self.encoder.decode(v))

    def calculate(self, x: float) -> float:
        raise NotImplementedError

    def get_optimal_x(self) -> float:
        raise NotImplementedError

    def apply(self, genotype):
        if self.is_caching:
            return self.cache_dict[genotype.tobytes()]
        return self.calculate(self.encoder.decode(genotype))

    def get_phenotype(self, chromosome):
        return self.encoder.decode(chromosome)

    @cache
    def get_optimal(self):
        optimal_genotype = self.encoder.encode(self.get_optimal_x())
        return Chromosome(0, optimal_genotype, self)

    def name(self) -> str:
        return f"{super().name()}_{self.encoder.name()}"


class Fx2(Freal):
    def calculate(self, x: float) -> float:
        return x**2

    def get_optimal_x(self) -> float:
        return self.encoder.upper_bound


class F5122subx2(Freal):
    def calculate(self, x: float) -> float:
        return 5.12**2 - x**2

    def get_optimal_x(self) -> float:
        return 0


class Fexp(Freal):
    def __init__(self, c: float, encoder: Encoder):
        self.c = c
        super().__init__(encoder)

    def calculate(self, x: float) -> float:
        return exp(self.c * x)

    def get_optimal_x(self) -> float:
        return self.encoder.upper_bound

    def name(self) -> str:
        return f"Fexp{self.c}_{self.encoder.name()}"


class FRastrigin(Freal):
    def calculate(self, x: float) -> float:
        a = 7
        return math.fabs(10 * math.cos(2 * math.pi * a) - a**2) + 10 * math.cos(2 * math.pi * x) - x**2

    def get_optimal_x(self) -> float:
        return 0


class FDebb2(Freal):
    def calculate(self, x: float) -> float:
        return math.exp(-2 * math.log(2) * np.square((x - 0.1) / 0.8)) * math.pow(math.sin(5 * math.pi * x), 6)

    def get_optimal_x(self) -> float:
        return 0.1


class FDebb4(Freal):
    def calculate(self, x: float) -> float:
        return math.exp(-2 * math.log(2) * np.square((x - 0.08) / 0.854)) * \
            math.pow(math.sin(5 * math.pi * (math.pow(x, 0.75) - 0.05)), 6)

    def get_optimal_x(self) -> float:
        return 0.08
