import random
from knapsack import knapsack_problem
"""
Here you will complete your Hill Climbing Implementation
You will need to use the Random Library to choose between fitness ties
DO NOT change the following random seed
"""
random.seed(534)

def best_neighbor(state, knapsack):
    """Here you will write a function to get the best neighbor from the current knapsack state
        Check all 30 single item changes (remove or add)
        if there is a tie, choose randomly using random.randint at the end

        Args:
            state - the knapsack chromosome to get a neighbor for
            knapsack - the knapsack object to get state values with

        Returns: best neighbor - a 30 item array if a best neighbor is found
                    if none is found, return an empty array or return None"""

    # keep a list of best neighbors to choose from randomly for ties
    poss_selections = []

    # keep a best value variable to keep track of what neighbor value is best.
    best_value = 0

    for i in range(30):
        # make a copy of the current knapsack to alter
        neighbor = state.copy()

        # alter copy at index i to create next neighbor
        if neighbor[i] == 0:
            neighbor[i] = 1
        else:
            neighbor[i] = 0

        # check if neighbor has greater value than current best value
        neighbor_value = knapsack.get_value(neighbor)

        # if it does, make this neighbor's value the best value and reset possible neighbors to only this one
        if neighbor_value > best_value:
            poss_selections = [neighbor]
            best_value = neighbor_value
        # if it has the same value as the best, add it to possible neighbors to choose from at the end
        elif neighbor_value == best_value:
            poss_selections.append(neighbor)

    # if the possible selections array is not empty, return a random neighbor from it
    if len(poss_selections) != 0:
        return poss_selections[random.randint(0,len(poss_selections)-1)]
    # if it is empty return None or empty array
    else:
        return None


def hill_climb():

    """Follow the pseudo-code in figure 4.2 of your book. """
    print("Hill-climbing implementation")

    # Start with Initial Empty Knapsack - use knapsack_problem object
    knapsack = knapsack_problem()
    current_state = knapsack.empty_sack()

    # initiate your fitness curve as empty array
    curve = []


    while (True):
        # append the value of the current state to your fitness curve for graphing later
        curve.append(knapsack.get_value(current_state))

        # get highest valued neighbor
        neighbor = best_neighbor(current_state, knapsack)

        # if value of neighbor is less than or equal to the value of the current state, return the current state
        if knapsack.get_value(neighbor) <= knapsack.get_value(current_state):
            return current_state, curve
        # if it is greater, the neighbor becomes the current state
        else:
            current_state = neighbor


# Prints results
if __name__ == "__main__":
    result, _ = hill_climb()
    print(result)
    print(knapsack_problem().get_value(result))


