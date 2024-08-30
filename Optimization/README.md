# Randomized Optimzation

**In order to explore random search, I have analyzed three algorithms and three problems which highlight the benefits and costs of each algorithm. Random Hill climbing (RHC), Simulated Annealing (SA), Genetic 
Algorithms (GA), and Mimic (MM) are analyzed for each problem. The first three of these algorithms are analyzed in use for finding neural network weights against gradient descent.**

See **cbeveridge3-analysis.pdf** for the full analysis


The code for each optimization problem and for neural networks is contained within its own file. All results for each problem can be 
reproduced with its corresponding python file. Running each file will produce figures identical to the ones in the analysis report for that algorithm, except for figure 4b which is
a combination of 2 figures. The grid search code is commented out in all files.

These figures are named by the code in the following:  

Figure1a - napsackit.png
Figure1b - napsacksize.png

Figure2a - randomrulecomp.png
Figure2b - randomrulesize.png

Figure3a - love7comp.png
Figure3b - love7size.png

Figure4a - neuralfit.png
Figure4b - neuralacc.png, neuralacc2.png

neural.py - Neural Network Code - Tic tac toe dataset
napsack.py - Knapsack problem Code
randomrule.py - Random Rule Problem code
love7.py - Loves 7s problem code


in the input folder has the dataset, the features are obvious in Tic-tac-toe, make sure to create the input folder in your directory as well!:
tic-tac-toe.csv - Tic-Tac-Toe Engame dataset

Modules used:

matplotlib
mlrose_hiive
timeit
numpy
sklearn
pandas
