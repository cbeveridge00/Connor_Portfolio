from sklearn.random_projection import SparseRandomProjection
from scipy.sparse import csr_matrix
import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import pinv
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
import statistics
from sklearn.cluster import FeatureAgglomeration
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot

#use pandas to get data in a nice format
game_data = pandas.read_csv("input/tic-tac-toe.csv")
heart_data = pandas.read_csv("input/heart.csv")

#labelencode the values with onehot encoding
x = game_data.iloc[:, 0:9]
y = game_data.iloc[:, 9]

heart_x = heart_data.iloc[:, 0:13]
heart_y = heart_data.iloc[:, 13]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

one_hot = pandas.get_dummies(heart_x['cp'])
heart_x = heart_x.drop('cp',axis = 1)
heart_x = heart_x.join(one_hot)

heart_x = StandardScaler().fit_transform(heart_x)
x = pandas.get_dummies(x)
x = StandardScaler().fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=200, train_size=758, random_state=212)
X_train_heart, X_test_heart, y_train_heart, y_test_heart = train_test_split(heart_x, heart_y, test_size=200,
                                                                            train_size=758, random_state=354)

#run all DRs on the heart dataset

#pca
pca = PCA(random_state=10)
principalComponents = pca.fit_transform(X_train_heart)
principalComponents = np.array(principalComponents)
principalComponents = principalComponents[:,:11]

#ICA
ica = FastICA(n_components=8, random_state=10, max_iter=3000)
ica_ = ica.fit_transform(X_train_heart)


#RP
rp = SparseRandomProjection(n_components=13, random_state=10)
rp_ = rp.fit_transform(X_train_heart)

#FA
fa = FeatureAgglomeration(n_clusters=3, linkage='average')
heart_red = fa.fit_transform(X_train_heart)



#run all DRs on the TicTacToe dataset

#pca
pca = PCA(random_state=10)
principalComponents_game = pca.fit_transform(X_train)
principalComponents = np.array(principalComponents)
principalComponents = principalComponents[:,:15]

#ICA
ica = FastICA(n_components=3, random_state=10)
ica_game = ica.fit_transform(X_train)

#RP
rp = SparseRandomProjection(n_components=21, random_state=10)
rp_game = rp.fit_transform(X_train)

#FA
fa = FeatureAgglomeration(n_clusters=3)
game_red = fa.fit_transform(X_train)


range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15]
#Run the Clustering algs

#Kmeans

score_kPCA = []
score_kPCA2 = []
score_PCA = []
score_PCA2 = []
savc = []
savc_5 = []

#PCA
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    clusterer2 = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=10)

    cluster_labels_game = clusterer.fit_predict(principalComponents_game)
    cluster_labels_heart = clusterer.fit_predict(principalComponents)

    cluster2_labels_game = clusterer2.fit_predict(principalComponents_game)
    cluster2_labels_heart = clusterer2.fit_predict(principalComponents)

    if n_clusters==2:
        savc = cluster_labels_heart
        savc2 = cluster_labels_game

        savc_em = cluster2_labels_heart
        savc2_em = cluster2_labels_game

    if n_clusters == 5:
        savc_5 = cluster2_labels_heart

    silhouette_avg_game = silhouette_score(principalComponents_game, cluster_labels_game)
    silhouette_avg_heart = silhouette_score(principalComponents, cluster_labels_heart)

    silhouette_avg_game2 = silhouette_score(principalComponents_game, cluster2_labels_game)
    silhouette_avg_heart2 = silhouette_score(principalComponents, cluster2_labels_heart)

    score_kPCA.append(silhouette_avg_game)
    score_kPCA2.append(silhouette_avg_heart)

    score_PCA.append(silhouette_avg_game2)
    score_PCA2.append(silhouette_avg_heart2)

print(score_PCA2)

#em

ytrain = y_train_heart.tolist()
score = 0
for i in range(len(savc_em)):
    if int(savc_em[i])==int(ytrain[i]):
        score += 1

print(score/758)

ytrain = y_train.tolist()
score = 0
for i in range(len(savc2_em)):
    if int(savc2_em[i])==int(ytrain[i]):
        score += 1

print(score/758)



ytrain = y_train_heart.tolist()
score = 0
for i in range(len(savc)):
    if int(savc[i])==int(ytrain[i]):
        score += 1

print(score/758)

ytrain = y_train.tolist()
score = 0
for i in range(len(savc2)):
    if int(savc2[i])==int(ytrain[i]):
        score += 1

print(score/758)




ica_score_k = []
ica_score_k2 = []
ica_score_em = []
ica_score2_em = []
savi = []
savi2 = []
savi_em = []
savi2_em = []
#ICA
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    clusterer2 = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=10)
    cluster_labels_game = clusterer.fit_predict(ica_game)
    cluster_labels_heart = clusterer.fit_predict(ica_)

    cluster2_labels_game = clusterer2.fit_predict(ica_game)
    cluster2_labels_heart = clusterer2.fit_predict(ica_)

    if n_clusters==2:
        savi = cluster_labels_heart
        savi2 = cluster_labels_game

        savi_em = cluster2_labels_heart
        savi2_em = cluster2_labels_game

    silhouette_avg_game = silhouette_score(ica_game, cluster_labels_game)
    silhouette_avg_heart = silhouette_score(ica_, cluster_labels_heart)

    silhouette_avg_game2 = silhouette_score(ica_game, cluster2_labels_game)
    silhouette_avg_heart2 = silhouette_score(ica_, cluster2_labels_heart)

    ica_score_k.append(silhouette_avg_game)
    ica_score_k2.append(silhouette_avg_heart)

    ica_score2_em.append(silhouette_avg_game2)
    ica_score_em.append(silhouette_avg_heart2)

'''

ytrain = y_train_heart.tolist()
score = 0
for i in range(len(savi_em)):
    if int(savi_em[i])==int(ytrain[i]):
        score += 1

print(score/758)

ytrain = y_train.tolist()
score = 0
for i in range(len(savi2_em)):
    if int(savi2_em[i])==int(ytrain[i]):
        score += 1

print(score/758)
'''

rp_score_k = []
rp_score_k2 = []
rp_score_em2 = []
rp_score_em = []
savr = []
savr2 = []
savr_em = []
savr2_em = []
#RP
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    clusterer2 = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=10)
    cluster_labels_game = clusterer.fit_predict(rp_game)
    cluster_labels_heart = clusterer.fit_predict(rp_)

    cluster2_labels_game = clusterer2.fit_predict(rp_game)
    cluster2_labels_heart = clusterer2.fit_predict(rp_)

    if n_clusters==2:
        savr = cluster_labels_heart
        savr2 = cluster_labels_game

        savr_em = cluster_labels_heart
        savr2_em = cluster_labels_game

    silhouette_avg_game = silhouette_score(rp_game, cluster_labels_game)
    silhouette_avg_heart = silhouette_score(rp_, cluster_labels_heart)

    silhouette_avg_game2 = silhouette_score(rp_game, cluster2_labels_game)
    silhouette_avg_heart2 = silhouette_score(rp_, cluster2_labels_heart)

    rp_score_k.append(silhouette_avg_game)
    rp_score_k2.append(silhouette_avg_heart)

    rp_score_em.append(silhouette_avg_game2)
    rp_score_em2.append(silhouette_avg_heart2)



ytrain = y_train_heart.tolist()
score = 0
for i in range(len(savr_em)):
    if int(savr_em[i])==int(ytrain[i]):
        score += 1

print(score/758)

ytrain = y_train.tolist()
score = 0
for i in range(len(savr2_em)):
    if int(savr2_em[i])==int(ytrain[i]):
        score += 1

print(score/758)



fa_score_k = []
fa_score_k2 = []
fa_score_em2 = []
fa_score_em = []
savf_em = []
savf2_em = []
savf = []
savf2 = []
#fa
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    clusterer2 = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=10)
    cluster_labels_game = clusterer.fit_predict(game_red)
    cluster_labels_heart = clusterer.fit_predict(heart_red)

    cluster2_labels_game = clusterer2.fit_predict(game_red)
    cluster2_labels_heart = clusterer2.fit_predict(heart_red)

    if n_clusters==2:
        savf = cluster_labels_heart
        savf2 = cluster_labels_game

        savf_em = cluster_labels_heart
        savf2_em = cluster_labels_game

    if n_clusters==9:
        savf2_9 = cluster2_labels_game

    silhouette_avg_game = silhouette_score(game_red, cluster_labels_game)
    silhouette_avg_heart = silhouette_score(heart_red, cluster_labels_heart)

    silhouette_avg_game2 = silhouette_score(game_red, cluster2_labels_game)
    silhouette_avg_heart2 = silhouette_score(heart_red, cluster2_labels_heart)

    fa_score_k.append(silhouette_avg_game)
    fa_score_k2.append(silhouette_avg_heart)

    fa_score_em.append(silhouette_avg_game2)
    fa_score_em2.append(silhouette_avg_heart2)



ytrain = y_train_heart.tolist()
score = 0
for i in range(len(savf_em)):
    if int(savf_em[i])==int(ytrain[i]):
        score += 1

print(score/758)

ytrain = y_train.tolist()
score = 0
for i in range(len(savf2_em)):
    if int(savf2_em[i])==int(ytrain[i]):
        score += 1

print(score/758)


fig8 = plt.figure(figsize=(12, 8))
ax2 = Axes3D(fig8, rect=[0, 0, 1, 1], elev=48, azim=134)

# Reorder the labels to have colors matching the cluster results

ax2.scatter(principalComponents[:, 0], principalComponents[:, 1], principalComponents[:, 2], c=savc, edgecolor='k')


ax2.w_xaxis.set_ticklabels([])
ax2.w_yaxis.set_ticklabels([])
ax2.w_zaxis.set_ticklabels([])
ax2.set_xlabel('P1')
ax2.set_ylabel('P2')
ax2.set_zlabel('P3')
ax2.set_title('K-means after PCA - Heart Disease')
ax2.dist = 11


fig8.savefig('fig8a.png')

fig8b = plt.figure(figsize=(12, 8))
ax2 = Axes3D(fig8b, rect=[0, 0, 1, 1], elev=48, azim=134)

# Reorder the labels to have colors matching the cluster results

ax2.scatter(principalComponents[:, 0], principalComponents[:, 1], principalComponents[:, 2], c=savc_5, edgecolor='k')


ax2.w_xaxis.set_ticklabels([])
ax2.w_yaxis.set_ticklabels([])
ax2.w_zaxis.set_ticklabels([])
ax2.set_xlabel('P1')
ax2.set_ylabel('P2')
ax2.set_zlabel('P3')
ax2.set_title('EM after PCA - Heart Disease')
ax2.dist = 11


fig8b.savefig('fig8b.png')


fig9a = plt.figure(figsize=(12, 8))
ax2 = Axes3D(fig9a, rect=[0, 0, 1, 1], elev=48, azim=134)

# Reorder the labels to have colors matching the cluster results

ax2.scatter(game_red[:, 0], game_red[:, 1], game_red[:, 2], c=savf2_9, edgecolor='k')


ax2.w_xaxis.set_ticklabels([])
ax2.w_yaxis.set_ticklabels([])
ax2.w_zaxis.set_ticklabels([])
ax2.set_xlabel('P1')
ax2.set_ylabel('P2')
ax2.set_zlabel('P3')
ax2.set_title('EM after PCA - Heart Disease')
ax2.dist = 11


fig9a.savefig('fig9a.png')



