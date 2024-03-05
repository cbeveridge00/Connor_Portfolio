
board1 = [
    [7, 8, 0, 4, 0, 0, 1, 2, 0],
    [6, 0, 0, 0, 7, 5, 0, 0, 9],
    [0, 0, 0, 6, 0, 1, 0, 7, 8],
    [0, 0, 7, 0, 4, 0, 2, 6, 0],
    [0, 0, 1, 0, 5, 0, 9, 3, 0],
    [9, 0, 4, 0, 6, 0, 0, 0, 5],
    [0, 7, 0, 3, 0, 0, 0, 1, 2],
    [1, 2, 0, 0, 0, 7, 4, 0, 0],
    [0, 4, 9, 2, 0, 6, 0, 0, 7]
]

board2 = [
    [6, 5, 9, 0, 1, 0, 2, 8, 0],
    [1, 0, 0, 0, 5, 0, 0, 3, 0],
    [2, 0, 0, 8, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 3, 5, 0, 7, 0],
    [8, 0, 0, 9, 0, 0, 0, 0, 2],
    [0, 0, 3, 0, 7, 8, 6, 4, 0],
    [3, 0, 2, 0, 0, 9, 0, 0, 4],
    [0, 0, 0, 0, 0, 1, 8, 0, 0],
    [0, 0, 8, 7, 6, 0, 0, 0, 0]
]

board3 = [
    [0, 0, 0, 5, 3, 4, 0, 8, 0],
    [0, 8, 0, 0, 1, 0, 4, 0, 0],
    [0, 2, 0, 8, 0, 0, 0, 7, 1],
    [8, 0, 0, 0, 6, 0, 0, 5, 0],
    [4, 0, 0, 0, 0, 5, 8, 3, 0],
    [6, 3, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 3, 0, 0],
    [0, 0, 0, 0, 7, 0, 0, 0, 0],
    [0, 1, 6, 2, 0, 0, 0, 0, 0]
]


def show_board(board):
    """
    Prints the Puzzle in a nice format
    """

    if not board:
        print("fail")

    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("-------------------------")

        for j in range(9):
            if j % 3 == 0 and j != 0:
                print(" |  ", end="")

            if j == 8:
                print(board[i][j])
            else:
                print(str(board[i][j]) + " ", end="")


def select_unassigned_variable(board):
    """
    Finds the next empty square on the sudoku board. For grading consistency,
    search for empty square from left to right, top to bottom

    :param board: the current board state in the search
    :return: the (row, column) tuple of selected unassigned square, or return False if the board is complete (no empty
        squares)
    """

    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return i, j

    return False


def check_validity(board, number, position):
    """
    Checks if a number added to a certain position is valid for a given board given the rules of sudoku.

    :param board: the board state to check
    :param number: the added number (1-9) to check
    :param position: the position on the board the number has been added; a tuple (row, col)
    :return: True or False as to whether the number addition is valid
    """

    # Check row
    for i in range(9):
        if board[position[0]][i] == number and position[1] != i:
            return False

    # Check column
    for i in range(9):
        if board[i][position[1]] == number and position[0] != i:
            return False

    # Check box
    x = position[1] // 3
    y = position[0] // 3

    for i in range(y*3, y*3 + 3):
        for j in range(x * 3, x*3 + 3):
            if board[i][j] == number and (i, j) != position:
                return False

    return True


def backtrack(board):
    """
    Here you will complete the backtracking CSP search algorithm.

    :param board: the starting sudoku board to solve
    :return: The completed sudoku puzzle, or False is no solution exists

    """

    # first select the next empty square. If there are none, return the board as it is a solution

    position = select_unassigned_variable(board)
    if not position:
        return True
    else:
        row, col = position

    # iterate through each possible value (1-9)
    for number in range(1, 10):
        # if the current value (1-9) is valid for the selected empty square, then make that empty square now the new
        # value. For instance it could go from 0 (meaning empty) to a 5
        # Make sure to edit a deepcopy of the board, and not the initial board!
        if check_validity(board, number, position):

            board[row][col] = number

            result = backtrack(board)

            if result:
                return True

            board[row][col] = 0

    return False

if __name__ == "__main__":
    backtrack(board1)
    show_board(board1)
