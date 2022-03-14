import random

CROSSOVER_PROBABILITY = 0.8


def trap(genes, fitness_counted_so_far, height):
    if genes.__len__() == 3:
        # print genes
        if genes[0] == 1 and genes[1] == 1 and genes[2] == 1:
            # print "first"
            return fitness_counted_so_far + (3 ** height)  # contribution to fitness
        else:
            if genes[0] == 0 and genes[1] == 0 and genes[2] == 0:
                # print "second"
                return fitness_counted_so_far + 0.8*(3 ** height)  # contribution to fitness
            else:
                if (genes[0] == 1 and genes[1] == 0 and genes[2] == 0) \
                        or (genes[0] == 0 and genes[1] == 1 and genes[2] == 0) \
                        or (genes[0] == 0 and genes[1] == 0 and genes[2] == 1):
                    # print "third"
                    return fitness_counted_so_far + 0.4 * (3 ** height)  # contribution to fitness

                else:
                    if (genes[0] == 1 and genes[1] == 1 and genes[2] == 0) \
                            or (genes[0] == 0 and genes[1] == 1 and genes[2] == 1) \
                            or (genes[0] == 1 and genes[1] == 0 and genes[2] == 1):       # contribution to fitness is 0
                        # print "forth"
                        return fitness_counted_so_far
                    else:
                        if genes[0] == -1 or genes[1] == -1 or genes[2] == -1:
                            # print "this   " + str(fitness_counted_so_far)
                            return fitness_counted_so_far

    else:                                                                   # low level
        fitness = []
        i = 0
        # print genes
        # print str(genes.__len__()) + "  gene"
        # print fitness_counted_so_far
        new = fitness_counted_so_far
        while i < genes.__len__():
            if genes[i] == 1 and genes[i+1] == 1 and genes[i+2] == 1:
                new = new + (3 ** height)     # contribution to fitness
                fitness.append(1)                                                   # the mapping
                # print "one"
            else:
                if genes[i] == 0 and genes[i + 1] == 0 and genes[i + 2] == 0:
                    new = new + (3 ** height)  # contribution to fitness
                    fitness.append(0)                                                # the mapping
                    # print "two"
                else:
                    if (genes[i] == 1 and genes[i + 1] == 0 and genes[i + 2] == 0)\
                            or (genes[i] == 0 and genes[i + 1] == 1 and genes[i + 2] == 0)\
                            or (genes[i] == 0 and genes[i + 1] == 0 and genes[i + 2] == 1):
                        new = new + 0.5*(3 ** height)  # contribution to fitness
                        fitness.append(-1)
                        # print "three"
                    else:
                        if (genes[i] == 1 and genes[i + 1] == 1 and genes[i + 2] == 0) \
                                or (genes[i] == 0 and genes[i + 1] == 1 and genes[i + 2] == 1) \
                                or (genes[i] == 1 and genes[i + 1] == 0 and genes[i + 2] == 1):
                                # contribution to fitness is 0
                            fitness.append(-1)
                            # print "four"
                        else:
                            if genes[i] == -1 or genes[i+1] == -1 or genes[i+2] == -1:
                                fitness.append(-1)
                                # print "five"
            # print str(i) + "    i"
            i = i + 3
            # print "here" + str(i)
        new_height = height + 1
        return trap(fitness, new, new_height)


class Chromosome:

    '''represents a candid solution'''
    def __init__(self, string_length):
        self._genes = []
        self._fitness = 0
        i = 0
        while i < string_length:
            if random.random() >= 0.5:
                self._genes.append(1)
            else:
                self._genes.append(0)
            i = i + 1

    def get_genes(self):
        return self._genes

    def set_genes(self, chromosome):
        self._genes = chromosome

    def get_fitness(self):
        init_fitness = 0
        height = 0
        fitness = trap(self._genes, init_fitness, height)
        return fitness

    def __str__(self):
        return self._genes.__str__()


class Population:
    '''respresents a population of candidate solutions'''
    def __init__(self, size, string_size):
        self._chromosomes = []
        i = 0
        while i < size:
            self._chromosomes.append(Chromosome(string_size))
            i += 1

    def get_chromosomes(self):
        return self._chromosomes

    def set_chromosome(self, chromosome):
        self._chromosomes.append(chromosome)


class GeneticAlgorithm:
    # where mutation and crossover happens

    def __init__(self, population_size, string_length, mutation_prob):
        self._pop_size = population_size
        self._string_length = string_length
        self._mutation_prob = mutation_prob

    def evolve(self, pop):
        evolved_population = self._mutate_population(self._crossover_population(pop))
        return evolved_population

    def _crossover_chromosomes(self, chromosome1, chromosome2):
        child1 = Chromosome(0)
        child2 = Chromosome(0)
        father = chromosome1
        mother = chromosome2
        index = random.randint(1, self._string_length - 2)
        child1.set_genes(father.get_genes()[:index] + mother.get_genes()[index:])
        child2.set_genes(mother.get_genes()[:index] + father.get_genes()[index:])
        return child1, child2

    def _mutate_chromosome(self, chromosome):
        for i in range(self._string_length):
            if random.random() < self._mutation_prob:
                if random.random() < 0.5:
                    chromosome.get_genes()[i] = 1
                else:
                    chromosome.get_genes()[i] = 0

    def _crossover_population(self, pop):
        choose_parents = pop.get_chromosomes()
        parents_size = self._pop_size / 2
        max_fitness = sum(chromosome.get_fitness() for chromosome in choose_parents)
        # print "max fitness   " + str(max_fitness)
        first_index = random.uniform(0, 1/self._pop_size)
        for j in range(parents_size):
            parent1 = choose_parents[int(first_index)]
            second_index = first_index + (1 / parents_size)
            parent2 = choose_parents[int(second_index)]

            '''
            pick = random.uniform(0, max_fitness)       # using a roulett wheel
            current = 0
            for i in range(choose_parents.__len__()):
                current += choose_parents[i].get_fitness()
                if current > pick:
                    parent1 = choose_parents[i]
                    index1 = i
                    break

            pick = random.uniform(0, max_fitness)  # using a roulett wheel
            current = 0
            for k in range(choose_parents.__len__()):
                current += choose_parents[k].get_fitness()
                if current > pick:
                    parent2 = choose_parents[k]
                    index2 = k
                    break
                '''
            child1, child2 = self._crossover_chromosomes(parent1, parent2)     # compare new children with parents
            child1_parent1_difference = 0
            child1_parent2_difference = 0
            child2_parent1_difference = 0
            child2_parent2_difference = 0
            for counter in range(self._string_length):
                if child1.get_genes()[counter] == parent1.get_genes()[counter]:
                    child1_parent1_difference = child1_parent1_difference + 1
                else:
                    if child1.get_genes()[counter] == parent2.get_genes()[counter]:
                        child1_parent2_difference = child1_parent2_difference + 1
                    else:
                        if child2.get_genes()[counter] == parent1.get_genes()[counter]:
                            child2_parent1_difference = child2_parent1_difference + 1
                        else:
                            if child2.get_genes()[counter] == parent2.get_genes()[counter]:
                                child2_parent2_difference = child2_parent2_difference + 1
            if child1_parent1_difference >= child1_parent2_difference:     # child1 is closer to parent1
                if child1.get_fitness() >= parent1.get_fitness():                                  # child is better than the parent
                    pop.get_chromosomes()[int(first_index)] = child1
                else:
                    pop.get_chromosomes()[int(first_index)] = parent2
            else:                                                                   # child is closer to parent2
                if child1.get_fitness() >= parent2.get_fitness():                   # child is better than the parent
                    pop.get_chromosomes()[int(first_index)] = child1
                else:
                    pop.get_chromosomes()[int(first_index)] = parent2

            if  child2_parent1_difference >= child2_parent2_difference:      # child2 is closer to parent1
                if child2.get_fitness() >= parent1.get_fitness():                                  # child is better than the parent
                    pop.get_chromosomes()[int(second_index)] = child2
                else:
                    pop.get_chromosomes()[int(second_index)] = parent1
            else:                                                                           # child is closer to parent1
                if child2.get_fitness() >= parent2.get_fitness():                                  # child is better than the parent
                    pop.get_chromosomes()[int(second_index)] = child2
                else:
                    pop.get_chromosomes()[int(second_index)] = parent2
            first_index = second_index + (1 / parents_size)
        # print_population(crossover_pop, 100)
        return pop

    def _mutate_population(self, pop):
        for i in range(self._pop_size/2):
            self._mutate_chromosome(pop.get_chromosomes()[i])
        return pop


def print_population(pop, gen_number):
    print"\n------------------------------------"
    print "generation #", gen_number, "|fittest chromosome fitness:", pop.get_chromosomes()[0].get_fitness()
    print "------------------------------------"
    i = 0
    for x in pop.get_chromosomes():
        print "chromosome #", i, " :", x, "fitness", x.get_fitness()
        i += 1


def GA(population_size, strings_length, target, number_of_iterations, mutation_probability):
    genetic = GeneticAlgorithm(population_size, strings_length, mutation_probability)
    population = Population(population_size, strings_length)    # the random first generation
    population.get_chromosomes().sort(key=lambda x: x.get_fitness(), reverse=True)  # sorting the generation by fitness
    # print_population(population, 0)
    iter_counter = 0
    while (iter_counter < number_of_iterations) and (population.get_chromosomes()[0].get_fitness() != target):
        new_generation = genetic.evolve(population)
        new_generation.get_chromosomes().sort(key=lambda x: x.get_fitness(), reverse=True)
        population = Population(0, 0)
        for i in range(population_size):
            population.get_chromosomes().append(new_generation.get_chromosomes()[i])
        iter_counter += 1
        # print_population(population, iter_counter)
    return population.get_chromosomes()[0], iter_counter,


def scenario():
    print "Fitness : hierarchical 3-bit trap"
    print "*** scenario #1: Population size->10, string size->27, iteration number->50"
    population_size = 10
    strings_length = 27
    target = 27
    number_of_iterations = 50
    mutation_probability = 1/population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations, mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest/5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation/5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest
    print "fittest: ", best
    print '\n'

    print "*** scenario #2: Population size->20, string size->27, iteration number->50"
    population_size = 20
    strings_length = 27
    target = 27
    number_of_iterations = 50
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest
    print "fittest: ", best
    print '\n'

    print "*** scenario #3: Population size->50, string size->27, iteration number->50"
    population_size = 50
    strings_length = 27
    target = 27
    number_of_iterations = 50
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest
    print "fittest: ", best
    print '\n'

    print "*** scenario #4: Population size->10, string size->27, iteration number->100"
    population_size = 10
    strings_length = 27
    target = 27
    number_of_iterations = 100
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest
    print "fittest: ", best
    print '\n'

    print "*** scenario #5: Population size->20, string size->27, iteration number->100"
    population_size = 20
    strings_length = 27
    target = 27
    number_of_iterations = 100
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest
    print "fittest: ", best
    print '\n'

    print "*** scenario #6: Population size->50, string size->27, iteration number->100"
    population_size = 50
    strings_length = 27
    target = 27
    number_of_iterations = 100
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest
    print "fittest: ", best
    print '\n'

    print "*** scenario #7: Population size->10, string size->27, iteration number->200"
    population_size = 10
    strings_length = 27
    target = 27
    number_of_iterations = 200
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest
    print "fittest: ", best
    print '\n'

    print "*** scenario #8: Population size->20, string size->27, iteration number->200"
    population_size = 20
    strings_length = 27
    target = 27
    number_of_iterations = 200
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest
    print "fittest: ", best
    print '\n'

    print "*** scenario #9: Population size->50, string size->27, iteration number->200"
    population_size = 50
    strings_length = 27
    target = 27
    number_of_iterations = 200
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest
    print "fittest: ", best
    print '\n'

    print "********************** string with 81 bits *************************"
    print "Fitness : hierarchical 3-bit trap"
    print "*** scenario #1: Population size->10, string size->81, iteration number->50"
    population_size = 10
    strings_length = 81
    target = 81
    number_of_iterations = 50
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest
    print "fittest: ", best
    print '\n'

    print "*** scenario #2: Population size->20, string size->81, iteration number->50"
    population_size = 20
    strings_length = 81
    target = 81
    number_of_iterations = 50
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest
    print "fittest: ", best
    print '\n'

    print "*** scenario #3: Population size->50, string size->81, iteration number->50"
    population_size = 50
    strings_length = 81
    target = 81
    number_of_iterations = 50
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest
    print "fittest: ", best
    print '\n'

    print "*** scenario #4: Population size->10, string size->81, iteration number->100"
    population_size = 10
    strings_length = 81
    target = 81
    number_of_iterations = 100
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest
    print "fittest: ", best
    print '\n'

    print "*** scenario #5: Population size->20, string size->81, iteration number->100"
    population_size = 20
    strings_length = 81
    target = 81
    number_of_iterations = 100
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest
    print "fittest: ", best
    print '\n'

    print "*** scenario #6: Population size->50, string size->81, iteration number->100"
    population_size = 50
    strings_length = 81
    target = 81
    number_of_iterations = 100
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest
    print "fittest: ", best
    print '\n'

    print "*** scenario #7: Population size->10, string size->81, iteration number->200"
    population_size = 10
    strings_length = 81
    target = 81
    number_of_iterations = 200
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest
    print "fittest: ", best
    print '\n'

    print "*** scenario #8: Population size->20, string size->81, iteration number->200"
    population_size = 20
    strings_length = 81
    target = 81
    number_of_iterations = 200
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest
    print "fittest: ", best
    print '\n'

    print "*** scenario #9: Population size->50, string size->81, iteration number->200"
    population_size = 50
    strings_length = 81
    target = 81
    number_of_iterations = 200
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest
    print "fittest: ", best


scenario()






