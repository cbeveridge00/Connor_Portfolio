import tkinter as tk
from PIL import ImageTk, Image
from checkersBackend import *
from adversarial import *
import random
import time

class CheckersUI (tk.Frame):
    STICKY = tk.N + tk.S + tk.E + tk.W

    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.grid()
        self.gameMenu()

    def gameMenu(self):
        self.start = tk.Frame(self)
        self.start.grid(row=0)


        self.okButton = tk.Button(self, text="Free Play", height = 5,
          width = 40, padx=10, pady=10, command=self.gameStartupDialog)
        self.okButton.grid(row=0, column=0)

        img = ImageTk.PhotoImage(Image.open("wayne-community-college-logo.png"))
        self.panel = tk.Label(self, image=img)
        self.panel.image = img
        self.panel.grid(row=0, column=1)

        self.okButton = tk.Button(self, text="Play Vs. Minimax Depth = 3 Bot", height = 5,
          width = 40, padx=10, pady=10, command=self.beginGameBots)
        self.okButton.grid(row=1, column=0)

        self.okButton = tk.Button(self, text="Play Vs. Alpha-Beta Depth = 5 Bot", height = 5,
          width = 40, padx=10, pady=10, command=self.beginGameAlpha)
        self.okButton.grid(row=1, column=1)

        self.okButton = tk.Button(self, text="Play Vs. Custom Bot", height = 5,
          width = 40, padx=10, pady=10, command=self.beginGameCustom)
        self.okButton.grid(row=2, column=0)

        self.okButton = tk.Button(self, text="Test Mode", height = 5,
          width = 40, padx=10, pady=10, command=self.testMode)
        self.okButton.grid(row=2, column=1)


    def gameStartupDialog(self):


        for widget in self.winfo_children():
            widget.destroy()



        self.start = tk.Frame(self)



        self.start.grid(row=0)

        self.p1Name = tk.StringVar()
        self.p2Name = tk.StringVar()
        self.timer = tk.BooleanVar()

        tk.Label(self.start, text="Player 1 Name").grid(padx=50, pady=30,
            row=0, column=0)
        tk.Entry(self.start, textvariable=self.p1Name).grid(padx=50, pady=30,
            row=0, column=1)

        tk.Label(self.start, text="Player 2 Name").grid(padx=50, pady=30,
            row=1, column=0)
        tk.Entry(self.start,  textvariable=self.p2Name).grid(padx=50, pady=30,
            row=1, column=1)

        self.okButton = tk.Button(self, text="Ok", height = 3,
          width = 20, command=self.beginGame)
        self.okButton.grid(padx=10, pady=10, row=3, column = 0)



    def drawBots(self):
        self.boardCanvas.destroy()
        self.boardCanvas = tk.Canvas(self.game, width=560, height=560)

        self.boardCanvas.grid(row=2, column=0, columnspan=2)
        self.boardCanvas.create_image((280, 280), image=self.backgroundPhoto)

        for i in range(8):
            for j in range(8):
                if self.positions[i][j] != None:
                    self.boardCanvas.create_image((35 + 70 * i, 35 + 70 * j),
                                                  image=self.checkersPieceDict[self.positions[i][j]])

        self.boardCanvas.bind("<Button-1>", self.clickBoardBots)



    def draw(self):
        self.boardCanvas.destroy()
        self.boardCanvas = tk.Canvas(self.game, width=560, height=560)
        self.boardCanvas.grid(row=2, column=0, columnspan=2)
        self.boardCanvas.create_image((280, 280), image=self.backgroundPhoto)
        
        for i in range(8):
            for j in range(8):
                if self.positions[i][j] != None:
                    self.boardCanvas.create_image((35 + 70*i, 35 + 70*j),
                                                  image=self.checkersPieceDict[self.positions[i][j]])
        
        self.boardCanvas.bind("<Button-1>", self.clickBoard)

    def testMode(self):

        moves = minimax(initialBoard(), max_depth=1)

        if moves == "Not Implemented":
            print("Minimax Not Implemented")
        else:

            minimax_score = 0
            # Minimax 3 vs random bot
            for i in range(10):
                self.positions = initialBoard()

                self.beginGameHelperBots()

                self.drawBots()
                self.player1Name = "Random Bot"
                self.player2Name = "Minimax Bot"

                self.playerTurnLabel = tk.Label(self.game, text="* " + self.player1Name)
                self.playerTurnLabel.grid(row=0, column=0)
                self.playerTurnLabel2 = tk.Label(self.game, text=self.player2Name)
                self.playerTurnLabel2.grid(row=0, column=1)

                self.boardCanvas.after(100, self.boardCanvas.update())

                while True:
                    self.turn = 0

                    # random (player 0) goes first
                    jmpDetectLst = jumpDetection(self.positions, 0)
                    if len(jmpDetectLst) != 0:
                        jmpLst = allJumpPositions(self.positions, 0)
                        (x, y, newX, newY) = jmpLst[random.randint(0, len(jmpLst) - 1)]
                        delX = int((newX + x) / 2.0)
                        delY = int((newY + y) / 2.0)
                        self.positions[delX][delY] = None
                        self.positions[newX][newY] = self.positions[x][y]
                        self.positions[x][y] = None

                        convertToKing(self.positions)

                        jmplst2 = jumpPositions(self.positions[newX][newY], newX, newY, self.positions)
                        x = newX
                        y = newY
                        while len(jmplst2) != 0:
                            (newX, newY) = jmplst2[random.randint(0, len(jmplst2) - 1)]
                            delX = int((newX + x) / 2.0)
                            delY = int((newY + y) / 2.0)
                            self.positions[delX][delY] = None
                            self.positions[newX][newY] = self.positions[x][y]
                            self.positions[x][y] = None

                            convertToKing(self.positions)

                            jmplst2 = jumpPositions(self.positions[newX][newY], newX, newY, self.positions)
                            x = newX
                            y = newY

                        self.drawBots()
                        self.boardCanvas.after(200, self.boardCanvas.update())

                    else:
                        mvlst = allMovePositions(self.positions)
                        if mvlst is None or len(mvlst) == 0:
                            minimax_score += 1
                            break

                        (x, y, newX, newY) = mvlst[random.randint(0, len(mvlst) - 1)]
                        self.positions[newX][newY] = self.positions[x][y]
                        self.positions[x][y] = None

                        convertToKing(self.positions)

                        self.drawBots()
                        self.boardCanvas.after(200, self.boardCanvas.update())

                    self.turn = 1
                    moves = minimax(self.positions, player=1)


                    if moves is None or len(moves) == 0:
                        break

                    for move in moves:
                        (x, y, newX, newY) = move

                        if abs(newX - x) > 1 or abs(newY - y) > 1:
                            delX = int((newX + x) / 2.0)
                            delY = int((newY + y) / 2.0)
                            self.positions[delX][delY] = None
                            self.positions[newX][newY] = self.positions[x][y]
                            self.positions[x][y] = None
                        else:
                            self.positions[newX][newY] = self.positions[x][y]
                            self.positions[x][y] = None

                        convertToKing(self.positions)

                        self.drawBots()
                        self.boardCanvas.after(200, self.boardCanvas.update())

            print("Minimax won " + str(minimax_score) + "/10 games")

        moves = alpha_beta(initialBoard(), max_depth=1)

        if moves == "Not Implemented":
            print("Alpha-beta Not Implemented")
        else:
            # Alpha Beta vs Minimax
            alpha_beta_score = 0
            for i in range(10):
                self.positions = initialBoard()

                self.beginGameHelperBots()

                self.drawBots()
                self.player1Name = "Random Bot"
                self.player2Name = "Alpha-Beta Bot"

                self.playerTurnLabel = tk.Label(self.game, text="* " + self.player1Name)
                self.playerTurnLabel.grid(row=0, column=0)
                self.playerTurnLabel2 = tk.Label(self.game, text=self.player2Name)
                self.playerTurnLabel2.grid(row=0, column=1)

                self.boardCanvas.after(100, self.boardCanvas.update())

                while True:
                    self.turn = 0

                    # random (player 0) goes first
                    jmpDetectLst = jumpDetection(self.positions, 0)
                    if len(jmpDetectLst) != 0:
                        jmpLst = allJumpPositions(self.positions, 0)
                        (x, y, newX, newY) = jmpLst[random.randint(0, len(jmpLst) - 1)]
                        delX = int((newX + x) / 2.0)
                        delY = int((newY + y) / 2.0)
                        self.positions[delX][delY] = None
                        self.positions[newX][newY] = self.positions[x][y]
                        self.positions[x][y] = None

                        convertToKing(self.positions)

                        self.drawBots()
                        self.boardCanvas.after(200, self.boardCanvas.update())

                        jmplst2 = jumpPositions(self.positions[newX][newY], newX, newY, self.positions)
                        x = newX
                        y = newY
                        while len(jmplst2) != 0:
                            (newX, newY) = jmplst2[random.randint(0, len(jmplst2) - 1)]
                            delX = int((newX + x) / 2.0)
                            delY = int((newY + y) / 2.0)
                            self.positions[delX][delY] = None
                            self.positions[newX][newY] = self.positions[x][y]
                            self.positions[x][y] = None

                            convertToKing(self.positions)

                            self.drawBots()
                            self.boardCanvas.after(200, self.boardCanvas.update())

                            jmplst2 = jumpPositions(self.positions[newX][newY], newX, newY, self.positions)
                            x = newX
                            y = newY

                    else:
                        mvlst = allMovePositions(self.positions)
                        if len(mvlst) == 0:
                            alpha_beta_score += 1
                            break

                        (x, y, newX, newY) = mvlst[random.randint(0, len(mvlst) - 1)]
                        self.positions[newX][newY] = self.positions[x][y]
                        self.positions[x][y] = None

                        convertToKing(self.positions)

                        self.drawBots()
                        self.boardCanvas.after(200, self.boardCanvas.update())

                    self.turn = 1
                    moves = alpha_beta(self.positions, player=1)

                    if moves is None or len(moves) == 0:
                        break

                    for move in moves:
                        (x, y, newX, newY) = move

                        if abs(newX - x) > 1 or abs(newY - y) > 1:
                            delX = int((newX + x) / 2.0)
                            delY = int((newY + y) / 2.0)
                            self.positions[delX][delY] = None
                            self.positions[newX][newY] = self.positions[x][y]
                            self.positions[x][y] = None
                        else:
                            self.positions[newX][newY] = self.positions[x][y]
                            self.positions[x][y] = None

                        convertToKing(self.positions)

                        self.drawBots()
                        self.boardCanvas.after(200, self.boardCanvas.update())

            print("AlphaBeta won " + str(alpha_beta_score) + "/10 games")

            self.resignGame()

        return



    def beginGameHelperBots(self):

        for widget in self.winfo_children():
            widget.destroy()

        self.game = tk.Frame(self)
        self.game.grid(row=0)


        self.backgroundPhoto = tk.PhotoImage(file="diagram.gif")
        self.checkersPieceDict = dict()
        photo = tk.PhotoImage(file="wm.gif")
        self.checkersPieceDict[Piece(0)] = photo
        photo = tk.PhotoImage(file="bm.gif")
        self.checkersPieceDict[Piece(1)] = photo

        photo = tk.PhotoImage(file="wk.gif")
        self.checkersPieceDict[Piece(0, True)] = photo

        photo = tk.PhotoImage(file="bk.gif")
        self.checkersPieceDict[Piece(1, True)] = photo

        self.boardCanvas = tk.Canvas(self.game, width=560, height=560)

        self.boardCanvas.grid(row=2, column=0, columnspan=2)
        self.boardCanvas.create_image((280, 280), image=self.backgroundPhoto)
        self.boardCanvas.bind("<Button-1>", self.clickBoardBots)

        self.statusLabel = tk.Label(self.game, text="")
        self.statusLabel.grid(row=3, column=0, columnspan=2)

        tk.Button(self.game, text="Resign", command=self.resignGame).grid(
            row=4, column=0)


        self.selected = False

    def beginGameHelper(self):
        self.okButton.grid_forget()
        self.start.grid_forget()

        self.game = tk.Frame(self)
        self.game.grid(row=0)        

        self.backgroundPhoto = tk.PhotoImage(file="diagram.gif")
        self.checkersPieceDict = dict()
        photo = tk.PhotoImage(file = "wm.gif")
        self.checkersPieceDict[Piece(0)] = photo
        photo = tk.PhotoImage(file = "bm.gif")
        self.checkersPieceDict[Piece(1)] = photo

        photo = tk.PhotoImage(file = "wk.gif")
        self.checkersPieceDict[Piece(0, True)] = photo

        photo = tk.PhotoImage(file = "bk.gif")
        self.checkersPieceDict[Piece(1, True)] = photo


        self.boardCanvas = tk.Canvas(self.game, width=560, height=560)
        self.boardCanvas.grid(row=2, column=0, columnspan=2)
        self.boardCanvas.create_image((280, 280), image=self.backgroundPhoto)
        self.boardCanvas.bind("<Button-1>", self.clickBoard)


        self.statusLabel = tk.Label(self.game, text="")
        self.statusLabel.grid(row = 3, column = 0, columnspan =2)

        tk.Button(self.game, text="Resign", command=self.resignGame).grid(
            row=4, column=0)


        self.selected = False
        self.drawOffered = False

    def beginGameCustom(self):
        self.beginGameHelperBots()
        self.positions = initialBoard()
        self.drawBots()

        self.player1Name = "Custom Bot"
        self.player2Name = "Student"

        self.playerTurnLabel = tk.Label(self.game, text="* " + self.player1Name)
        self.playerTurnLabel.grid(row=0, column=0)
        self.playerTurnLabel2 = tk.Label(self.game, text=self.player2Name)
        self.playerTurnLabel2.grid(row=0, column=1)

        self.boardCanvas.after(1000, self.boardCanvas.update())
        self.turn = 0
        self.botType = "custom"
        self.moveCustom()


    def beginGameAlpha(self):
        self.beginGameHelperBots()
        self.positions = initialBoard()
        self.drawBots()

        self.player1Name = "AlphaBeta Bot"
        self.player2Name = "Student"

        self.playerTurnLabel = tk.Label(self.game, text="* " + self.player1Name)
        self.playerTurnLabel.grid(row=0, column=0)
        self.playerTurnLabel2 = tk.Label(self.game, text=self.player2Name)
        self.playerTurnLabel2.grid(row=0, column=1)

        self.boardCanvas.after(1000, self.boardCanvas.update())
        self.turn = 0
        self.botType = "ab"
        self.moveAlpha()


    def beginGameBots(self):

        self.beginGameHelperBots()
        self.positions = initialBoard()
        self.drawBots()


        self.player1Name = "MiniMax Bot"
        self.player2Name = "Student"

        self.playerTurnLabel = tk.Label(self.game, text="* " + self.player1Name)
        self.playerTurnLabel.grid(row=0, column=0)
        self.playerTurnLabel2 = tk.Label(self.game, text=self.player2Name)
        self.playerTurnLabel2.grid(row=0, column=1)

        self.boardCanvas.after(1000, self.boardCanvas.update())
        self.turn = 0
        self.botType = "m3"
        self.moveM3()



    def beginGame(self):


        self.beginGameHelper()

        self.positions = initialBoard()

        self.draw()

        if (self.p1Name.get() == ""):
            self.player1Name = "Player 1"
        else:
            self.player1Name = self.p1Name.get()

        if (self.p2Name.get() == ""):
            self.player2Name = "Player 2"
        else:
            self.player2Name = self.p2Name.get()

        self.playerTurnLabel = tk.Label(self.game, text= "* "+ self.player1Name)
        self.playerTurnLabel.grid(row=0, column=0)
        self.playerTurnLabel2 = tk.Label(self.game, text=self.player2Name)
        self.playerTurnLabel2.grid(row=0, column=1)
        
        self.turn = 0


    def clickBoardBots(self, event):

        if self.turn == 0:
            return

        if noMoveDetection(self.positions, self.turn):
            self.statusLabel["text"] = "No possible moves, you have lost"
            print("no possible moves")
            self.resignGame()

        else:
            jmpDetectLst = jumpDetection(self.positions, 1)

            if self.selected == False:
                ptx, pty = pixelToInt(event.x, event.y)

                if (self.positions[ptx][pty] == None):
                    self.statusLabel["text"] = "No piece selected"


                else:
                    if (self.positions[ptx][pty].color != 1):
                        self.statusLabel["text"] = "Wrong color selected"


                    else:
                        s = set(jmpDetectLst)
                        if len(jmpDetectLst) != 0 and ((ptx, pty) not in s):

                            self.statusLabel["text"] = "Incorrect selection. You have to jump"
                        else:
                            self.selected = True
                            self.selectedPt = (ptx, pty)

                            self.statusLabel["text"] = str(self.selectedPt) + " selected"

            else:
                ptx, pty = pixelToInt(event.x, event.y)
                self.moveStudent(ptx, pty)


    def clickBoard(self, event):


        
        if noMoveDetection(self.positions, self.turn):
            self.statusLabel["text"] = "No possible moves, you have lost"

            self.resignGame()

        else:
            jmpDetectLst = jumpDetection(self.positions, self.turn)

            if self.selected == False:
                ptx, pty = pixelToInt(event.x, event.y)

                if (self.positions[ptx][pty] == None):
                    self.statusLabel["text"] = "No piece selected"

                
                else:
                    if (self.positions[ptx][pty].color != self.turn):
                        self.statusLabel["text"] = "Wrong color selected"


                    else:
                        s = set(jmpDetectLst)
                        if len(jmpDetectLst) != 0 and ((ptx, pty) not in s):

                            self.statusLabel["text"] = "Incorrect selection. You have to jump"
                        else:
                            self.selected = True
                            self.selectedPt = (ptx, pty)

                            self.statusLabel["text"] = str(self.selectedPt) +  " selected"
                        
            else:
                ptx, pty = pixelToInt(event.x, event.y)
                self.move(ptx, pty)
    
    def setPlayer1(self):
        


        self.playerTurnLabel["text"] = "* "+ self.player1Name
        self.playerTurnLabel2["text"] = self.player2Name
        self.selected = False
        self.statusLabel["text"] = ""
        self.turn = 0



    def setPlayer2(self):

        self.playerTurnLabel["text"] = self.player1Name
        self.playerTurnLabel2["text"] = "* " + self.player2Name
        self.selected = False
        self.statusLabel["text"] = ""
        self.turn = 1

    def moveCustom(self):

        self.drawBots()

        self.boardCanvas.after(500, self.boardCanvas.update())

        moves = custom_bot(self.positions)

        if moves == "Not Implemented":
            self.game.destroy()
            self.gameMenu()
            return

        return

    def moveAlpha(self):
        self.drawBots()

        self.boardCanvas.after(200, self.boardCanvas.update())

        moves = alpha_beta(self.positions, max_depth=7)

        self.botMoves(moves)
        return

    def moveM3(self):

        self.drawBots()

        self.boardCanvas.after(500, self.boardCanvas.update())

        moves = minimax(self.positions)

        self.botMoves(moves)
        return

    def botMoves(self, moves):
        if moves == "Not Implemented":
            self.game.destroy()
            self.gameMenu()
            return

        if moves is None or len(moves)==0:
            self.winGame()
            return

        for move in moves:
            (x, y, newX, newY) = move

            if abs(newX - x) > 1 or abs(newY - y) > 1:
                delX = int((newX + x) / 2.0)
                delY = int((newY + y) / 2.0)
                self.positions[delX][delY] = None
                self.positions[newX][newY] = self.positions[x][y]
                self.positions[x][y] = None
            else:
                self.positions[newX][newY] = self.positions[x][y]
                self.positions[x][y] = None

            convertToKing(self.positions)
            self.drawBots()
            self.boardCanvas.after(200, self.boardCanvas.update())

        self.setPlayer2()
        return

    def moveStudent(self, x2, y2):
        ptx, pty = self.selectedPt
        jmplst = jumpPositions(self.positions[ptx][pty], ptx, pty, self.positions)
        mvlst = movePositions(self.positions[ptx][pty], ptx, pty, self.positions)

        if len(jmplst) != 0:
            s = set(jmplst)
            if ((x2, y2) not in s):
                self.statusLabel["text"] = str(self.selectedPt) + " selected, you have to take the jump"
                return
            else:
                delX = int((x2 + ptx) / 2.0)
                delY = int((y2 + pty) / 2.0)
                self.positions[delX][delY] = None
                self.positions[x2][y2] = self.positions[ptx][pty]
                self.positions[ptx][pty] = None
                self.selected = True
                self.selectedPt = (x2, y2)

                convertToKing(self.positions)
                self.drawBots()

                jmplst2 = jumpPositions(self.positions[x2][y2], x2, y2, self.positions)

                if len(jmplst2) != 0:
                    self.statusLabel["text"] = str(self.selectedPt) + " selected, you must jump again"
                    return

                else:
                    if (noOpponentPieceDetection(self.positions, self.turn)):
                        self.winGame()
                        print("won game")


                    self.setPlayer1()
                    if self.botType == "m3":
                        self.moveM3()
                    elif self.botType == "ab":
                        self.moveAlpha()
                    else:
                        self.moveCustom()
        else:
            s = set(mvlst)
            if ((x2, y2) not in s):
                self.statusLabel["text"] = "Invalid move. Select a piece and try again"
                self.selected = False
                return
            else:
                self.positions[x2][y2] = self.positions[ptx][pty]
                self.positions[ptx][pty] = None

                convertToKing(self.positions)
                self.drawBots()


                self.setPlayer1()
                if self.botType == "m3":
                    self.moveM3()
                elif self.botType == "ab":
                    self.moveAlpha()
                else:
                    self.moveCustom()
            
    def move(self, x2, y2):
        ptx, pty = self.selectedPt
        jmplst = jumpPositions(self.positions[ptx][pty], ptx, pty, self.positions)
        mvlst = movePositions(self.positions[ptx][pty], ptx, pty, self.positions)



        if len(jmplst) != 0:
            s = set(jmplst)
            if ((x2, y2) not in s):
                self.statusLabel["text"] = str(self.selectedPt) +  " selected, you have to take the jump"
                return
            else:
                delX = int((x2+ptx)/2.0)
                delY = int((y2+pty)/2.0)
                self.positions[delX][delY] = None
                self.positions[x2][y2] = self.positions[ptx][pty]
                self.positions[ptx][pty] = None
                self.selected = True
                self.selectedPt = (x2, y2)

                convertToKing(self.positions)
                self.draw()

                jmplst2 = jumpPositions(self.positions[x2][y2], x2, y2, self.positions)
                
                if len(jmplst2) != 0:
                    self.statusLabel["text"] = str(self.selectedPt) +  " selected, you must jump again"
                    return

                else:
                    if (noOpponentPieceDetection(self.positions, self.turn)):
                        self.winGame()
                        print ("won game")

                    else:
                        if self.turn == 0:
                            self.setPlayer2()
                        else:
                            self.setPlayer1()
        else:
            s = set(mvlst)
            if ((x2, y2) not in s):
                self.statusLabel["text"] = "Invalid move. Select a piece and try again"
                self.selected = False
                return
            else:
                self.positions[x2][y2] = self.positions[ptx][pty]
                self.positions[ptx][pty] = None

                convertToKing(self.positions)
                self.draw()

                if self.turn == 0:
                    self.setPlayer2()
                else:
                    self.setPlayer1()
                

    def endGame(self):
        self.game.destroy()

        self.endFrame = tk.Frame(self)
        self.endFrame.grid(row = 0)
        self.endGameResult = tk.Label(self.endFrame, text= "")
        self.endGameResult.grid(row=0)
        tk.Button(self.endFrame, text="Main Menu", command=self.gameMenu).grid(row=1)
        tk.Button(self.endFrame, text="Exit", command=self.quit).grid(row=2)
    
    def endGameNew(self):
        self.endFrame.destroy()
        # self.loadButton.destroy()
        self.okButton.destroy()
        # self.timerFrame.destroy()
        self.start.destroy()
        
        self.gameStartupDialog()

    def resignGame(self):
        self.endGame()

        if self.turn == 0:
            winner = self.player2Name
            loser = self.player1Name
            
        else:
            winner = self.player1Name
            loser = self.player2Name

        self.endGameResult["text"] = winner + " won this game. " + loser + " lost."
        
    def winGame(self):
        self.endGame()

        if self.turn == 0:
            winner = self.player1Name
            loser = self.player2Name
            
        else:
            winner = self.player2Name
            loser = self.player1Name

        self.endGameResult["text"] = winner + " won this game. " + loser + " lost."


    def acceptDraw(self):
        self.endGame()
        self.endGameResult["text"] = "This game was a draw"

    def offerDraw(self):
        self.drawOffered = True
        self.offerDrawButton["state"] = tk.DISABLED



def pixelToInt(x, y):
    retx = 0
    retx_tot = 70
    
    rety = 0
    rety_tot = 70
    
    while (x > retx_tot and retx < 7):
        retx = retx + 1
        retx_tot = retx_tot + 70
        
    while (y > rety_tot and rety < 7):
        rety = rety + 1
        rety_tot = rety_tot + 70
        
    return (retx, rety)
    
def tkinter_main():
    root = tk.Tk()
    root.title('Wayne CC - Project 2: Checkers')

    root.configure(background='grey')
    app = CheckersUI(master=root)
    app.mainloop()

tkinter_main()
