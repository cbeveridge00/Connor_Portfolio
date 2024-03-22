
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

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


x = pandas.get_dummies(x)




range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15,20]
number_instance = [25,50,100,200,400,758]
best_score_g = []
best_score_h = []
best_kg = 0
best_kh = 0
best_tempg = 0
best_temph = 0
best_sog = 0
best_soh = 0
for num in number_instance:
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=200, train_size=num, random_state=212)
    X_train_heart, X_test_heart, y_train_heart, y_test_heart = train_test_split(heart_x, heart_y, test_size=200,
                                                                               train_size=num, random_state=354)
    best_tempg = 0
    best_temph = 0
#decide which k is best for k-means for these
    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels_game = clusterer.fit_predict(X_train)
        cluster_labels_heart = clusterer.fit_predict(X_train_heart)

        silhouette_avg_game = silhouette_score(X_train, cluster_labels_game)
        silhouette_avg_heart = silhouette_score(X_train_heart, cluster_labels_heart)

        if silhouette_avg_game > best_tempg:
            best_tempg = silhouette_avg_game
            if best_tempg > best_sog:
                best_sog = best_tempg
                best_kg = n_clusters

        if silhouette_avg_heart > best_temph:
            best_temph = silhouette_avg_heart
            if best_temph > best_soh:
                best_soh = best_temph
                best_kh = n_clusters

    best_score_g.append(best_tempg)
    best_score_h.append(best_temph)

print(best_score_g)
print(best_score_h)

fig3 = plt.figure(figsize=(14,2))
plt.plot(number_instance, best_score_g, 'g-o', label='TTT')
plt.plot(number_instance,best_score_h,'r-x', label='Heart Disease')
plt.xlim(0,800)
plt.ylim(0,.5)
plt.xlabel('Number of Instances')
plt.ylabel('Silhouette Score')
plt.legend()
fig3.savefig("fig1.png")

#do single optimal kmeans for each

clusterer = KMeans(n_clusters=14, random_state=10)
clusterer2 = KMeans(n_clusters=2, random_state=10)
cluster_labels_game = clusterer.fit_predict(X_train)

cluster_labels_heart = clusterer2.fit_predict(X_train_heart)
score = 0

ytrain = y_train_heart.tolist()

for i in range(len(cluster_labels_heart)):
    if int(cluster_labels_heart[i])!=int(ytrain[i]):
        score += 1

print(score/758)


# Plot the ground truth - heart
fig = plt.figure(figsize=(12, 9))
fig2 = plt.figure(figsize=(12, 9))
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=48, azim=134)
ax2 = Axes3D(fig2, rect=[0, 0, 1, 1], elev=48, azim=134)

X_train_heart = np.array(X_train_heart.values.tolist())
cluster_labels_heart = np.logical_not(cluster_labels_heart).astype(int)
# Reorder the labels to have colors matching the cluster results
ax.scatter(X_train_heart[:, 2], X_train_heart[:, 3], X_train_heart[:, 6], c=ytrain, edgecolor='k')
ax2.scatter(X_train_heart[:, 2], X_train_heart[:, 3], X_train_heart[:, 6], c=cluster_labels_heart, edgecolor='k')
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Blood Pressure')
ax.set_ylabel('Cholesterol')
ax.set_zlabel('Fasting Blood Sugar')
ax.set_title('Ground Truth')
ax.dist = 12

fig.savefig('gtheart.png')

ax2.w_xaxis.set_ticklabels([])
ax2.w_yaxis.set_ticklabels([])
ax2.w_zaxis.set_ticklabels([])
ax2.set_xlabel('Blood Pressure')
ax2.set_ylabel('Cholesterol')
ax2.set_zlabel('Fasting Blood Sugar')
ax2.set_title('K-Means Clusters')
ax2.dist = 12

fig2.savefig('clusterheart.png')

