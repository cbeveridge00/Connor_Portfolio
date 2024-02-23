import copy


def getMoves(player, board):
    """ gets all moves for a current player's turn from a board state

        Args:
            player (0 or 1): the current player; 0 is player one, 1 player two
            board: the board (a list) containing all the pieces (class Piece); rows of the list are columns of the board

        Returns:
            list of move(s):
                [[(x, y, newX, newY)...], [(x, y, newX, newY)...]...]: a list of lists of tuples of moves to make
                (more than one move can occur with double, triple, jumps)

                    x: current x location of piece on board (row index of board list)
                    y: current y location of piece on board (column index of board list)
                    newX: new x location of moved piece on board (row index of board list)
                    newY: new y location of moved piece on board (column index of board list)

    """

    jmpDetectLst = jumpDetection(board, player)

    if len(jmpDetectLst) != 0:
        return jumpActions(board, player)

    else:
        return moveActions(player, board)

def perform_action(action, board):
    """ performs a given move(s) on a board and returns the resulting board

        Args:
            action: the move(s) to be performed in the form:
                [(x, y, newX, newY)...]: a list of tuples of moves to make (more than one move can occur with
                double, triple jumps)

                    x: current x location of piece on board (row index of board list)
                    y: current y location of piece on board (column index of board list)
                    newX: new x location of moved piece on board (row index of board list)
                    newY: new y location of moved piece on board (column index of board list)

            board: the board (a list) containing all the pieces (class Piece); rows of the list are columns of the board

        Return:
            new_board = the resulting board after actions are performed
    """
    new_board = copy.deepcopy(board)
    for move in action:
        (x, y, newX, newY) = move

        if abs(newX - x) > 1 or abs(newY - y) > 1:
            delX = int((newX + x) / 2.0)
            delY = int((newY + y) / 2.0)
            new_board[delX][delY] = None
            new_board[newX][newY] = new_board[x][y]
            new_board[x][y] = None
        else:
            new_board[newX][newY] = new_board[x][y]
            new_board[x][y] = None

        convertToKing(new_board)

    return new_board

def boundsCheck(f):
    
    def retfunc(*args):
        xlst = f(*args)
        
        y = []
        for x in xlst:
            x1, x2 = x
            if x1 < 8 and x1 >= 0 and x2 < 8 and x2 >= 0:
                y.append(x)
        
        return y
        
    return retfunc


@boundsCheck
def surroundings(piece, x, y, board):
    ret = []
    if (piece.king):
        ret.append((x - 1, y - 1))
        ret.append((x + 1, y + 1))
        ret.append((x + 1, y - 1))
        ret.append((x - 1, y + 1))
        return ret
    
    else:
        if (piece.color == 0):
            ret.append((x + 1, y + 1))
            ret.append((x - 1, y + 1))
            return ret
        else:
            ret.append((x - 1, y - 1))
            ret.append((x + 1, y - 1))
            return ret 


@boundsCheck
def possibleJumps(piece, x, y, board):
    positions = surroundings(piece, x, y, board)
    
    ret = []
    
    for p in positions:
        i, j = p
        if board[i][j] != None:
            if board[i][j].color != piece.color:
                ret.append((x+2*(i - x), y + 2*(j - y)))
    
    return ret


def jmpTree(board, x, y, jmpOG):

    moves = []

    jmpLst = jumpPositions(board[x][y], x, y, board)


    if len(jmpLst) == 0:
        return [[jmpOG]]



    for jmp in jmpLst:

        #moves.append(jmpOG)


        boardCopy = copy.deepcopy(board)
        (newX, newY) = jmp
        delX = int((newX + x) / 2.0)
        delY = int((newY + y) / 2.0)
        boardCopy[delX][delY] = None
        boardCopy[newX][newY] = boardCopy[x][y]
        boardCopy[x][y] = None

        convertToKing(boardCopy)

        jmpBuild = jmpTree(boardCopy, newX, newY, (x, y, jmp[0], jmp[1]))

        for e in jmpBuild:
            e.insert(0, jmpOG)
            moves.append(e)


    return moves

def jumpActions(board, player):

    ret = []


    jmpLst = allJumpPositions(board, player)


    for jmp in jmpLst:


        boardCopy = copy.deepcopy(board)
        (x, y, newX, newY) = jmp
        delX = int((newX + x) / 2.0)
        delY = int((newY + y) / 2.0)
        boardCopy[delX][delY] = None
        boardCopy[newX][newY] = boardCopy[x][y]
        boardCopy[x][y] = None

        convertToKing(boardCopy)

        for e in jmpTree(boardCopy, newX, newY, jmp):
            ret.append(e)

    return ret


def moveActions(player, board):

    ret = []
    for x in range(8):
        for y in range(8):
            if board[x][y] != None and board[x][y].color == player:
                mvLst = movePositions(board[x][y], x, y, board)

                if len(mvLst) == 0:
                    pass
                else:
                    for mv in mvLst:
                        ret.append([(x, y, mv[0], mv[1])])
    return ret

def allJumpPositions(board, player):

    ret = []

    for x in range(8):
        for y in range(8):
            if board[x][y] != None and board[x][y].color == player:
                jmpLst = jumpPositions(board[x][y], x, y, board)

                if len(jmpLst) == 0:
                    pass
                else:
                    for jmp in jmpLst:
                        ret.append((x, y, jmp[0], jmp[1]))
    return ret

def jumpPositions(piece, x, y, board):
    positions = possibleJumps(piece, x, y, board)
    
    ret = []
    
    for p in positions:
        i, j = p

        if board[i][j] == None:
            ret.append((i, j))

    return ret

def allMovePositions(board):


    ret = []
    for x in range(8):
        for y in range(8):
            if board[x][y] != None and board[x][y].color == 0:
                mvLst = movePositions(board[x][y], x, y, board)

                if len(mvLst) == 0:
                    pass
                else:
                    for mv in mvLst:
                        ret.append((x, y, mv[0], mv[1]))
    return ret


def movePositions(piece, x, y, board):
    positions = surroundings(piece, x, y, board)

    ret = []

    for p in positions:
        i, j = p
        
        if board[i][j] == None:
            ret.append((i, j))


    return ret

def initialBoardTest():

    board = [[Piece(0), None, Piece(0), None, None, None, None, None],
             [None, Piece(1), None, None, None, None, None, None],
             [Piece(0), None, Piece(0), None, None, None, None, None],
             [None, Piece(0), None, None, None, None, None, None],
             [Piece(0), None, None, None, None, None, None, None],
             [None, Piece(0), None, None, None, None, None, None],
             [Piece(0), None, None, None, None, None, None, None],
             [None, Piece(0), None, None, None, None, None, None]]

    return board

def initialBoard():
    ret = []
    
    for i in range(8):
        ith_row = []
        ret.append(ith_row)
        
        for j in range(8):
            ret[i].append(None)

    
    for i in range(0, 4, 2):
        for j in range(0, 7, 2):
            ret[j][i] = Piece(0)
            ret[1+j][7-i] = Piece(1)
    

    for j in range(0, 7, 2):
        ret[1+j][1] = Piece(0)
        ret[j][6] = Piece(1)

    return ret


def convertToKing(board):
    for j in range(8):
        if board[j][0] != None:
            if board[j][0].color == 1 and board[j][0].king != True:
                board[j][0].king = True
                
        if board[j][7] != None:
            if board[j][7].color == 0 and board[j][7].king != True:
                board[j][7].king = True
                
    return

def noMoveDetection(board, turn):
    ret = True

    for x in range(8):
        for y in range(8):
            if board[x][y] != None and board[x][y].color == turn:
                mvLst = movePositions(board[x][y], x, y, board)
                jmpLst = jumpPositions(board[x][y], x, y, board)
                
                if len(mvLst) == 0 and len(jmpLst) == 0:
                    pass
                else:
                    ret = False
    return ret


def noOpponentPieceDetection(board, turn):
    ret = True
    
    if turn == 0:
        opponent = 1
    else:
        opponent = 0

    for x in range(8):
        for y in range(8):
            if board[x][y] != None and board[x][y].color == opponent:
                ret = False

    return ret

def jumpDetection(board, turn):
    ret = []

    for x in range(8):
        for y in range(8):
            if board[x][y] != None and board[x][y].color == turn:
                jmpLst = jumpPositions(board[x][y], x, y, board)
                
                if len(jmpLst) == 0:
                    pass
                else:
                    ret.append((x, y))
    return ret


class Piece:
    def __init__(self, color, king = False):
        self.color = color
        self.king = king
    
    def __eq__(self, other):
        if isinstance(other, Piece):
            return (self.color == other.color) and (self.king == other.king)
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)    


    def __hash__(self):
        return hash((self.color, self.king))
