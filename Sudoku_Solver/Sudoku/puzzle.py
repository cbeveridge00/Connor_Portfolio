import pygame


class Puzzle:

    def __init__(self, board):
        self.cubes = [[Square(board[i][j], i, j) for j in range(9)] for i in range(9)]
        self.model = None

    def get_board(self):
        """returns a 2D array of the board"""
        return [[self.cubes[i][j].value for j in range(9)] for i in range(9)]

    def update_model(self):
        """Updates the board"""
        self.model = [[self.cubes[i][j].value for j in range(9)] for i in range(9)]

    def place(self, val, row, col):
        """
        alters a value on the board

        :param val: the value to add/change
        :param row: the row of the board to alter
        :param col: the column of the board to alter
        """
        self.cubes[row][col].set(val)
        self.update_model()

    def draw(self, win):
        # Draw Puzzle Lines
        gap = 540 / 9
        for i in range(9+1):
            if i % 3 == 0 and i != 0:
                thick = 6
            else:
                thick = 1
            pygame.draw.line(win, (0, 0, 0), (0, i*gap), (540, i*gap), thick)
            pygame.draw.line(win, (0, 0, 0), (i * gap, 0), (i * gap, 540), thick)

        # Draw Cubes
        for i in range(9):
            for j in range(9):
                self.cubes[i][j].draw(win)


class Square:

    def __init__(self, value, row, col):
        self.value = value
        self.row = row
        self.col = col

    def draw(self, win):
        fnt = pygame.font.SysFont("IrisUPC", 40)

        gap = 540 / 9
        x = self.col * gap
        y = self.row * gap

        if self.value == 0:
            text = fnt.render('', True, (128, 128, 128))
            win.blit(text, (x+5, y+5))
        else:
            text = fnt.render(str(self.value), True, (0, 0, 0))
            win.blit(text, (x + (gap/2 - text.get_width()/2), y + (gap/2 - text.get_height()/2)))

    def set(self, val):
        self.value = val
