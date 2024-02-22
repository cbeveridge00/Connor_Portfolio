

class knapsack_problem:

    def __init__(self):
        self.weights = [6, 24, 5, 18, 1, 18, 28, 22, 18, 25, 17, 23, 21, 27, 1, 19, 11, 20, 24, 20, 12, 14, 22, 27, 2, 4, 22, 28, 19, 23]

        self.values = [17, 8, 11, 11, 5, 7, 19, 4, 20, 28, 10, 19, 18, 22, 7, 22, 12, 14, 7, 30, 16, 22, 27, 12, 16, 8, 4, 19, 30, 17]


    def empty_sack(self):
        "Provides an empty knapsack chromosome for starting algorithm runs"
        return [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    def check_weight(self, sack):
        """Checks if a current knapsack chromosome indicates a weight that the knapsack can hold

                Arg: a chromosome representation of a knapsack state
                Returns: True if the weight is not greater than 140, False otherwise.
        """

        weight = 0
        for i in range(30):
            weight += self.weights[i]*sack[i]

        if weight > 140:
            return False

        return True

    def get_value(self, sack):
        """ Calculates the fitness or total value of items in a knapsack. If the knapsack is too full
        or is empty, the value is 1 for value of the knapsack itself.

                Arg: a chromosome representation of a knapsack state
                Returns: the total value of items in the knapsack
        """

        if sack is None:
            return 1

        if not self.check_weight(sack):
            return 1

        value = 1
        for i in range(30):
            value += self.values[i] * sack[i]

        return value

