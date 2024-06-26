
## Supervised learning Project

In this project I did a deep analysis of a heart disease and tic-tac-toe dataset, comparing and constrasting different model types and their performance on these two datasets. The full analysis writeup can be found in Analysis.docx

The code for each algorithm is contained within its own file. An extra python file (curve.py) holds a fuction for plotting learning curves. All results for each algorithm can be 
reproduced with its corresponding python file. Running each file will produce figures identical to the ones in the analysis report for that algorithm. These figures are named after 
their name in the alaysis with 'a' indicating the first graph of the figure. For instance, figure 1 is 'fig1a.png' and 'fig1b.png'. Figure 2 is 'fig2a.png' and so on. Numeric results
and times from these anlysis are printed as output.  

- dt.py - Decision Tree Code
- ANN.py - Neural Network Code
- boost.py - Boosting code
- svm.py - Support Vector Machine Code
- knn.py - K-NN code

- curve.py - support for creating learning curves


In the input folder are the datasets, the features are obvious in Tic-tac-toe, but explanations for heart-disease features are given in the folder:
tic-tac-toe.csv - Tic-Tac-Toe Engame dataset
heart.csv - Heart-Disease dataset
heart_disease_features.txt - explains what features represent
