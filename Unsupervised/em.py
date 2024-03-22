
import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score
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
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=200, train_size=758, random_state=212)
X_train_heart, X_test_heart, y_train_heart, y_test_heart = train_test_split(heart_x, heart_y, test_size=200,
                                                                            train_size=758, random_state=354)


range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15]
cov_type = ['spherical', 'tied', 'diag', 'full']
best_covg = ''
best_covh = ''
best_kg = 0
best_kh = 0
best_g = 0
best_h = 0
for cov in cov_type:
#decide which component # is best for
    for n_clusters in range_n_clusters:
        clusterer = GaussianMixture(n_components=n_clusters, covariance_type=cov, random_state=10)
        cluster_labels_game = clusterer.fit_predict(X_train)
        cluster_labels_heart = clusterer.fit_predict(X_train_heart)

        silhouette_avg_game = silhouette_score(X_train, cluster_labels_game)
        silhouette_avg_heart = silhouette_score(X_train_heart, cluster_labels_heart)

        if silhouette_avg_game > best_g:
            best_g = silhouette_avg_game
            best_covg = cov
            best_kg = n_clusters

        if silhouette_avg_heart > best_h:
            best_h = silhouette_avg_heart
            best_covh = cov
            best_kh = n_clusters


print(best_g)
print(best_h)
print(best_covg)
print(best_covh)
print(best_kg)
print(best_kh)

clusterer2 = GaussianMixture(n_components=2, covariance_type='full', random_state=10)

cluster_labels_heart = clusterer2.fit_predict(X_train_heart)
score = 0

ytrain = y_train_heart.tolist()

for i in range(len(cluster_labels_heart)):
    if int(cluster_labels_heart[i])!=int(ytrain[i]):
        score += 1

print(score/758)


clusterer2 = GaussianMixture(n_components=2, covariance_type='full', random_state=10)


cluster_labels_heart = clusterer2.fit_predict(X_train_heart)

# Plot the ground truth - heart
fig2 = plt.figure(figsize=(12, 8))
ax2 = Axes3D(fig2, rect=[0, 0, 1, 1], elev=48, azim=134)

X_train_heart = np.array(X_train_heart)
# Reorder the labels to have colors matching the cluster results
cluster_labels_heart = np.logical_not(cluster_labels_heart).astype(int)

ax2.scatter(X_train_heart[:, 2], X_train_heart[:, 3], X_train_heart[:, 6], c=cluster_labels_heart, edgecolor='k')


ax2.w_xaxis.set_ticklabels([])
ax2.w_yaxis.set_ticklabels([])
ax2.w_zaxis.set_ticklabels([])
ax2.set_xlabel('Blood Pressure')
ax2.set_ylabel('Cholesterol')
ax2.set_zlabel('Fasting Blood Sugar')
ax2.set_title('EM Clusters')
ax2.dist = 11

fig2.savefig('emheart.png')



#Compare vs number of instances
clusterer2 = GaussianMixture(n_components=2, covariance_type='full', random_state=10)

number_instance = [25,50,100,200,400,758]

best_score_h = []
for num in number_instance:
    score = 0
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=200, train_size=num, random_state=212)
    X_train_heart, X_test_heart, y_train_heart, y_test_heart = train_test_split(heart_x, heart_y, test_size=200,
                                                                                train_size=num, random_state=354)

    cluster_labels_heart = clusterer2.fit_predict(X_train_heart)
    ytrain = y_train_heart.tolist()
    for i in range(len(cluster_labels_heart)):
        if int(cluster_labels_heart[i]) != int(ytrain[i]):
            score += 1

    s = score / num
    if s < .5:
        s = 1 - s
    print(s)
    best_score_h.append(s)


fig3 = plt.figure(figsize=(12,8))
#plt.plot(number_instance, best_score_g, 'g-o', label='TTT')
plt.plot(number_instance,best_score_h,'r-x', label='Heart Disease')
plt.xlim(0,800)
plt.ylim(.40,.75)
plt.xlabel('Number of Instances')
plt.ylabel('Accuracy to Ground Truth')
plt.legend()
fig3.savefig("fig3.png")