# Clustering and Dimensionality reduction Analysis

In this project I use the same datasasets as the previous supervised analysis that fairly represent the opposite ends of this spectrum of human 
understanding. I analyze how these datasets are handled by unsupervised methods including clustering and dimensionality reduction. I also use the resulting clusters as new features for the data in training a neural network.

The full analysis is under this folder as cbeveridge3-analysis.pdf.

The code and datasets can be found at the following github link.

The code for each section of the analysis is contained within its own file. All results for each problem can be 
reproduced with its corresponding python file. Running each file will produce figures identical to the ones in the analysis report for that algorithm, and print certain results to the terminal.
These figures are named by the code in the following:  

Figure1 - fig1.png

Figure2a - gtheart.png
Figure2b - clusterheart.png

Figure3a - emheart.png
Figure3b - fig3.png

Figure4a - pcagame.png
Figure4b - pcaheart.png

figure5a - icagame.png
figure5b - icaheart.png

figure6a - fagame.png
figure6b - faheart.png

figure8a - fig8a.png
figure8b - fig8b.png

figure9 - fig9a.png

Figure 10 - fig10.png

figure11a - fig11.png
figure11b - fig12.png

- kmeans.py - K Means Code
- em.py - EM Code
- pca.py - PCA code
- ica.py - ICA code
- rp.py - random projection code
- fa.py - feature agglomeration code
- CafterDr.py - Clustering on Reduced data problem code
- neural.py - Running Reduced Data on Neural Network Code
- final_neural.py - Running Clustering on Reduced Data as new Features Code

in the input folder has the dataset, the features are obvious in Tic-tac-toe, make sure to create the input folder in your directory as well!:
- tic-tac-toe.csv - Tic-Tac-Toe Engame dataset
- heart.csv - Heart Disease Data

Modules used:

matplotlib
timeit
numpy
sklearn
pandas
mpl_toolkits.mplot3d



You may want to comment some things out when you test each problem as gridsearches can take awhile. 
