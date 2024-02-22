import random
from knapsack import knapsack_problem
import numpy

"""
Here you will complete your Genetic Algorithm Implementation
You will need to use the Random Library to choose the random successor.
DO NOT change the following random seed
"""
random.seed(534)
numpy.random.seed(534)


def generate_population(knapsack):
    """
    Generate an initial population of 50 empty (all 0s) knapsack chromosomes and return them in an array

    :param: the knapsack object
    :return: 2D array of entire population
    """

    # initiate output array
    pop = []

    # for each of 50 knapsacks
    for k in range(300):
        # append the individual to the population array
        pop.append(knapsack.empty_sack())

    # return the population array
    return pop


def reproduce(parent1, parent2):
    """
    This will take 2 parent chromosomes, split them at the same index, and combine the left side at the split point of
    parent 1 and the right side of the split point of parent 2. A new child chromosome will result

    :param parent1: The first parent chromosome to combine
    :param parent2: The second parent chromosme to combine
    :return: The child
    """

    # use random.randint to get a random split point (think hard about the indexes for each parent, you should be able
    # to have a parent split where the parent 1 part is only index 0 and where the parent 2 part is only index 29).
    # These split points are possible indexes to split at in the 30 item array

    indx = random.randint(1, 29)

    parent1Part = parent1[:indx]
    parent2Part = parent2[indx:]

    # combine the 2 parent chromosome parts and return the resulting child chromosome (should be of length 30)
    return parent1Part + parent2Part


def mutate(child):
    """
    mutate a single child according to the mutation rate (.01). Thus, every item in the knapsack has a 1% chance of
    being added or removed (0 to 1 or 1 to 0).
    Use random.random

    :param child: the child to mutate
    :return: the mutated child
    """

    # for every item in the child chromosome, mutate it with 10% probability. Use random.random

    for i in range(30):
        rand = random.random()
        if rand < .01:
            if child[i] == 0:
                child[i] = 1
            else:
                child[i] = 0

    return child


def weighted_choice(population, weights):
    """
    Picks 2 parents based on their values. Higher valued knapsacks have a higher chance of being selected.
     - Use numpy.random.choice()
     - If all weights are 0 as in first iteration, all probabilities should be 1/300, otherwise a divide by 0 occurs
     - otherwise probabilities for each item are weight/(sum of weights)
     - make sure to use replace=False in numpy.random.choice()

    :param population:
    :param weights:
    :return: a tuple of 2 parents
    """

    # create an array of numbers 1-50 (# of individuals in the population) which will be passed to numpy.random.choice()
    # for selection these selections will be the indexes to select parents from the population

    # calculate probability of selection for each item, if all weights are zero, each is 1/30
    tot = sum(weights)
    if tot == 0:
        probs = [1 / 300 for _ in range(300)]
    else:
        probs = [weights[i] / tot for i in range(300)]

    # get random weighted choices which are indexes of individuals in the population
    ind1, ind2 = numpy.random.choice([*range(300)], 2, False, probs)

    # return population members at the indexes returned by numpy.random.choice
    return population[ind1], population[ind2]


def get_best_individual(population, knapsack):
    """
    This has been done for you, simply returns the best valued individual of a population

    :param population:
    :param knapsack:
    :return: best valued individual knapsack chromosome
    """

    best_value = 0
    best_individual = []
    for j in range(len(population)):
        value = knapsack.get_value(population[j])
        if value > best_value:
            best_value = value
            best_individual = population[j]

    return best_individual


def genetic():
    """
    This is the main function of the genetic algorithm. Follow the pseudocode in figure 4.8 of your book.
    Run for 80 iterations

    :return: the best valued knapsack in the population after 80 iterations
    """
    print("Genetic Algorithm Implementation")

    # Create a Knapsack_problem object for getting values
    knapsack = knapsack_problem()

    # First, generate your initial population from your generate_population function
    population = generate_population(knapsack)

    # initiate your fitness curve as empty array
    curve = []

    # Run 80 iterations
    for _ in range(80):

        # append the value of the best individual to your fitness curve for graphing later
        curr_best_val = knapsack.get_value(get_best_individual(population, knapsack))
        curve.append(curr_best_val)

        # create an array with weights for each individual in the population
        weights = []
        for individual in population:
            weights.append(knapsack.get_value(individual))

        # initiate new population array (empty)
        new_pop = []

        # for every individual in the population, pick 2 parents (weighted using weighted_choice function), reproduce,
        # mutate, and then replace old population with new one

        for _ in range(len(population)):
            parent1, parent2 = weighted_choice(population, weights)

            child = reproduce(parent1, parent2)
            child = mutate(child)

            new_pop.append(child)

        population = new_pop

    # Return the best individual (highest value) of the final population and the curve as a tuple

    # append the value of the best individual from your final population to your fitness curve for graphing later
    curr_best_val = knapsack.get_value(get_best_individual(population, knapsack))
    curve.append(curr_best_val)

    return get_best_individual(population, knapsack), curve


if __name__ == "__main__":
    result, _ = genetic()
    print(result)
    print(knapsack_problem().get_value(result))
