from puzzle import Puzzle
import pygame
import sys
from game import select_unassigned_variable, check_validity

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


def redraw_window(win, board):
    """Draws the newly added numbers on the board"""

    win.fill((255, 255, 255))

    # Draw Puzzle and board
    board.draw(win)


def main():
    """
    Main code to run the pygame visual puzzle
    """
    pygame.font.init()
    win = pygame.display.set_mode((540, 600))
    pygame.display.set_caption("Sudoku")
    board = Puzzle(board1)

    backtrack(board, win)

    run = True
    while run:
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                run = False


def backtrack(puzzle, win):
    """
    Here you will complete the backtracking CSP search algorithm to work with pygame.

    :param puzzle: the starting sudoku Puzzle object to solve. This is not a 2D array!
    :param win: the pygame window object
    :return: The completed sudoku Puzzle object, or False is no solution exists

    """
    # Do not edit the following

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # YOUR CODE HERE

    # first select the next empty square. If there are none, return the board as it is a solution
    position = select_unassigned_variable(puzzle.get_board())
    if not position:
        return True
    else:
        row, col = position

    # iterate through each possible value (1-9)
    for number in range(1, 10):
        # if the current value (1-9) is valid for the selected empty square, then make that empty square now the new
        # value. For instance it could go from 0 (meaning empty) to a 5
        # Make sure to edit a deepcopy of the board, and not the initial board!
        if check_validity(puzzle.get_board(), number, position):

            puzzle.place(number, row, col)

            redraw_window(win, puzzle)
            pygame.display.update()

            result = backtrack(puzzle, win)

            if result:
                return True

            puzzle.place(0, row, col)
            redraw_window(win, puzzle)
            pygame.display.update()

    return False


main()
