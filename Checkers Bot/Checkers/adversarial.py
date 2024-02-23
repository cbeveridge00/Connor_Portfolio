from checkersBackend import getMoves, perform_action
import copy

def evaluate_board(player, board):
    """outputs a score for a given board state

        Args:
            player (0 or 1): the current player; 0 is player one, 1 player two
            board: the board (a list) containing all the pieces (class Piece); rows of the list are columns of the board

        Returns:
            score: a numeric utility score

    """
    score = 0

    for x in range(8):
        for y in range(8):
            if board[x][y] != None:
                if board[x][y].color == player:
                    if board[x][y].king:
                        score += 3
                    else:
                        score += 1
                else:
                    if board[x][y].king:
                        score -= 3
                    else:
                        score -= 1



    return score

def isGameEnd(board):
    """
    checks if the a game state has been won by either player.
    An easy way to determine this is to check if each player has any available moves

    Args:
         board: the board state to check
    Returns:
        result: 0 if player 1 has won, 1 if player 2 has won, and -1 if the game has not ended
            draw
    """

    '''
    playerOneMoves = getMoves(0, board)
    playerTwoMoves = getMoves(1, board)


    if len(playerOneMoves) == 0:
        return 1
    elif len(playerTwoMoves) == 0:
        return 0
        
    '''
    player1count = False
    player2count = False
    for x in range(8):
        for y in range(8):
            if board[x][y] != None:
                if board[x][y].color == 0:
                    player1count = True
                else:
                    player2count = True


    if not player2count:
        return 0

    if not player1count:
        return 1

    return -1


def minimax(board, max_depth=3, player=0):
    """Implementation of the minimax algorithm.

    Do not change function parameters!

        Args:
            board: the board (a list) containing all the pieces (class Piece); rows of the list are columns of the board
            depth: the depth of the search tree to search to
            player (0 or 1): the current player; 0 is player one, 1 player two

        Returns:
            move(s):
                [(x, y, newX, newY)...]: a list of tuples of moves to make (more than one move can occur with double,
                triple, jumps)

                    x: current x location of piece on board (row index of board list)
                    y: current y location of piece on board (column index of board list)
                    newX: new x location of moved piece on board (row index of board list)
                    newY: new y location of moved piece on board (column index of board list)
    """

    value, move = max_value(player, board, 0, max_depth)

    return move

# get max val for minimax
def max_value(player, board, depth, max_depth):
    """expand a max node of the minimax search tree

        Args:
            player (0 or 1): the current player who initiated the minimax algorithm
            board: the board (a list) containing all the pieces (class Piece); rows of the list are columns of the board
            depth: the current depth of the search tree to search to
            max_depth: the depth in the search tree to search to

        Returns:
            utility, move(s)

                utility: the utility score of a move
                move: [(x, y, newX, newY)...]: a list of tuples of moves to make (more than one move can occur with
                double, triple jumps)

                    x: current x location of piece on board (row index of board list)
                    y: current y location of piece on board (column index of board list)
                    newX: new x location of moved piece on board (row index of board list)
                    newY: new y location of moved piece on board (column index of board list)
            """

    # If game is at a terminal state, return board evaluation or end game utility and Null move
    if depth == max_depth:
        return evaluate_board(player, board), None

    gameEnd = isGameEnd(board)
    if gameEnd == player:
        return 999, None
    elif gameEnd != -1:
        return -999, None


    best_val, best_move = -9999, None

    for a in getMoves(player, board):
        new_board = perform_action(a, board)

        new_value, newMove = min_value(player, new_board, depth+1, max_depth)

        if new_value > best_val:
            best_val = new_value
            best_move = a

    return best_val, best_move

# get min val for minimax
def min_value(player, board, depth, max_depth):
    """expand a min node of the minimax search tree

        important!: this function looks at the moves of the opposing player, not the current player!

        Args:
            player (0 or 1): the current player who initiated the minimax algorithm
            board: the board (a list) containing all the pieces (class Piece); rows of the list are columns of the board
            depth: the current depth of the search tree to search to
            max_depth: the max depth in the search tree to search to

        Returns:
            utility, move(s)

                utility: the utility score of a move
                move: [(x, y, newX, newY)...]: a list of tuples of moves to make (more than one move can occur with
                double, triple jumps)

                    x: current x location of piece on board (row index of board list)
                    y: current y location of piece on board (column index of board list)
                    newX: new x location of moved piece on board (row index of board list)
                    newY: new y location of moved piece on board (column index of board list)
            """

    if player==0:
        min_player = 1
    else:
        min_player = 0

    if depth == max_depth:
        return evaluate_board(player, board), None

    gameEnd = isGameEnd(board)
    if gameEnd == player:
        return 999, None
    elif gameEnd != -1:
        return -999, None

    best_val, best_move = 9999, None

    for a in getMoves(min_player, board):
        new_board = perform_action(a, board)

        new_value, newMove = max_value(player, new_board, depth + 1, max_depth)

        if new_value < best_val:
            best_val = new_value
            best_move = a

    return best_val, best_move



def alpha_beta(board, max_depth=5, player=0, alpha=float('-inf'), beta=float('inf')):
    """Implementation of the alphabeta algorithm.

    Do not change function parameters!
        Args:
            board: the board (a list) containing all the pieces (class Piece); rows of the list are columns of the board
            player (0 or 1): the current player who initiated the alpha-beta algorithm
            max_depth: the depth in the search tree to search to
            alpha (float): Alpha value for pruning
            beta (float): Beta value for pruning


        Returns:
            move(s):
                [(x, y, newX, newY)...]: a list of tuples of moves to make (more than one move can occur with double,
                     triple jumps)

                    x: current x location of piece on board (row index of board list)
                    y: current y location of piece on board (column index of board list)
                    newX: new x location of moved piece on board (row index of board list)
                    newY: new y location of moved piece on board (column index of board list)
        """

    value, move = ab_max_value(player, board, 0, max_depth, alpha, beta)
    return move
    #return "Not Implemented"

# get max val for minimax
def ab_max_value(player, board, depth, max_depth, alpha, beta):
    """expand a min node of the alphaBeta search tree

        Args:
            player (0 or 1): the current player who initiated the minimax algorithm
            board: the board (a list) containing all the pieces (class Piece); rows of the list are columns of the board
            depth: the current depth of the search tree to search to
            max_depth: the max depth in the search tree to search to
            alpha (float): Alpha value for pruning
            beta (float): Beta value for pruning

        Returns:
            utility, move(s):

                utility: the utility score of a move
                move: [(x, y, newX, newY)...]: a list of tuples of moves to make (more than one move can occur with
                double, triple jumps)

                    x: current x location of piece on board (row index of board list)
                    y: current y location of piece on board (column index of board list)
                    newX: new x location of moved piece on board (row index of board list)
                    newY: new y location of moved piece on board (column index of board list)
            """


    # If game is at a terminal state, return board evaluation or end game utility and Null move
    if depth == max_depth:
        return evaluate_board(player, board), None

    gameEnd = isGameEnd(board)

    if gameEnd == player:
        return 999, None
    elif gameEnd != -1:
        return -999, None

    best_val, best_move = -9999, None

    for a in getMoves(player, board):

        new_board = perform_action(a, board)

        new_value, newMove = ab_min_value(player, new_board, depth + 1, max_depth, alpha, beta)

        if new_value > best_val:
            best_val = new_value
            best_move = a
            alpha = max(alpha, best_val)

        if best_val >= beta:
            return best_val, best_move


    return best_val, best_move

# get min val for alpha-beta
def ab_min_value(player, board, depth, max_depth, alpha, beta):
    """expand a min node of the alphaBeta search tree

        important!: this function looks at the moves of the opposing player, not the current player!

        Args:
            player (0 or 1): the current player who initiated the minimax algorithm
            board: the board (a list) containing all the pieces (class Piece); rows of the list are columns of the board
            depth: the current depth of the search tree to search to
            max_depth: the max depth in the search tree to search to
            alpha (float): Alpha value for pruning
            beta (float): Beta value for pruning

        Returns:
            utility, move(s):

                utility: the utility score of a move
                move: [(x, y, newX, newY)...]: a list of tuples of moves to make (more than one move can occur with
                double, triple jumps)

                    x: current x location of piece on board (row index of board list)
                    y: current y location of piece on board (column index of board list)
                    newX: new x location of moved piece on board (row index of board list)
                    newY: new y location of moved piece on board (column index of board list)
            """

    if player==0:
        min_player = 1
    else:
        min_player = 0

    # If game is at a terminal state, return board evaluation or end game utility and Null move
    if depth == max_depth:
        return evaluate_board(player, board), None

    gameEnd = isGameEnd(board)

    if gameEnd == player:
        return 999, None
    elif gameEnd != -1:
        return -999, None

    best_val, best_move = 9999, None

    for a in getMoves(min_player, board):
        new_board = perform_action(a, board)

        new_value, newMove = ab_max_value(player, new_board, depth + 1, max_depth, alpha, beta)

        if new_value < best_val:
            best_val = new_value
            best_move = a
            beta = min(beta, best_val)

        if best_val <= alpha:
            return best_val, best_move

    return best_val, best_move


def custom_bot(board, max_depth=5, player=0, alpha=float('-inf'), beta=float('inf')):
    """Implementation of the a custom bot for competition.

    Do not change function parameters!
        Args:
            board: the board (a list) containing all the pieces (class Piece); rows of the list are columns of the board
            player (0 or 1): the current player who initiated the alpha-beta algorithm
            max_depth: the depth in the search tree to search to
            alpha (float): Alpha value for pruning
            beta (float): Beta value for pruning


        Returns:
            move(s):
                [(x, y, newX, newY)...]: a list of tuples of moves to make (more than one move can occur with double,
                     triple jumps)

                    x: current x location of piece on board (row index of board list)
                    y: current y location of piece on board (column index of board list)
                    newX: new x location of moved piece on board (row index of board list)
                    newY: new y location of moved piece on board (column index of board list)
        """


    return "Not Implemented"



