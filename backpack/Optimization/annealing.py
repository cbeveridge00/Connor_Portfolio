import random
from knapsack import knapsack_problem
import math

"""
Here you will complete your Simulated Annealing Implementation
You will need to use the Random Library to choose the random successor.
DO NOT change the following random seed
"""
random.seed(534)

def schedule(time):

    """
    Uses an exponential decay rate to get the temperature for each iteration
    :param time: The current iteration number
    :return: the scheduled temperature
    """

    return 50 * pow(.998, time)

def get_successor(state):

    """
    chooses a random neighbor of the current knapsack state
    :param state: the knapsack state to get a random neighbor for
    :return: the randomly selected neighbor
    """

    # Choose a random item in the knapsack to alter (add or remove) using random.randint
    index = random.randint(0, 29)

    # make a copy of the state to alter
    sack = state.copy()

    # alter this copy at the chosen index; add or remove item depending if it is absent or present
    if sack[index] == 0:
        sack[index] = 1
    else:
        sack[index] = 0

    # return the altered state
    return sack

def sim_annealing():

    """
    Follow the pseudo-code in figure 4.5 of your book. Note there is a Typo in this figure! See the README


    :return: the final optimized knapsack state
    """
    print("Simulated Annealing implementation")

    # Start with Initial Empty Knapsack - use knapsack_problem object
    knapsack = knapsack_problem()
    current_state = knapsack.empty_sack()

    # initiate your fitness curve as empty array
    curve = []

    # start at time = 0, if you use a for loop, you will need around 373,000 iterations to reach T = 0
    for time in range(1000000):
        # append the value of the current state to your fitness curve for graphing later
        curve.append(knapsack.get_value(current_state))

        # set the temp according to a geometric decay
        t = schedule(time)

        # if T reaches 0, return the current state
        if t == 0:
            return current_state, curve

        # get a random successor/neighbor
        successor = get_successor(current_state)

        # compute difference in value between neighbor - current state
        delE = knapsack.get_value(successor) - knapsack.get_value(current_state)

        # if this difference is greater than 0, the successor becomes the current state
        # Otherwise, the successor becomes the current state with probability e^delE/t - use math.exp and random.random
        if delE > 0:
            current_state = successor
        else:
            prob = math.exp(delE/t)
            rand = random.random()

            if rand < prob:
                current_state = successor


if __name__ == "__main__":
    result, _ = sim_annealing()
    print(result)
    print(knapsack_problem().get_value(result))