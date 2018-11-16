import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import homogeneity_score

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

def pre_proc():
    #df1 = pd.read_csv("all_mean_in_class.csv")
    #df2 = pd.read_csv("training_set_media_geral.csv")
    #df3 = pd.concat([df1, df2]).drop(['Unnamed: 0'], axis = 1).to_csv("d1_train_test_merge.csv", index = False)
    
    df3 = pd.read_csv("aps_clustering.csv")
    df3 = df3.drop(['ab_000','bn_000','bo_000','bp_000','bq_000','br_000','cr_000'], axis = 1)
    Y = df3['class']
    X = df3.drop(['class'], axis = 1)
    
    X_scaled = StandardScaler().fit_transform(X)
    X_scaled_tran = pd.DataFrame(data = X_scaled, columns = X.columns.values).to_csv("X_scaled1.csv", index = False)



def dbscan(X, At):
    # Compute DBSCAN using Iris dataset
    db = DBSCAN(eps=0.3, metric ="precomputed").fit(At)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'
    
        class_member_mask = (labels == k)
    
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=14)
    
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=6)
    
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
    
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))

    
def elbow(X):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
        print(i)
    
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.show()

def kmean(X, y):
    kmeans = KMeans(n_clusters = 2)
    y_kmeans = kmeans.fit_predict(X.values)
    print("SSE: ", kmeans.inertia_)
    X = X.values
    plt.figure(figsize=(8,6))
    #Visualising the clusters
    x0 = X[y_kmeans == 0, 0]
    x1 = X[y_kmeans == 0, 1]
    labels = kmeans.labels_.astype(np.int)
    x01 = X[y_kmeans == 1, 0]
    x11 = X[y_kmeans == 1, 1]
    
    plt.scatter(x0, x1, c = 'red', label = '0',alpha = 0.5, s=10)
    plt.scatter(x01, x11, c = 'blue', label = '1', alpha = 0.5, marker = 'x', s=10)
    
    #Plotting the centroids of the clusters
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], c = 'green', label = 'Centroids')
    plt.legend()

    #print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, y_kmeans), samples = 1000)
    #print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(Y.values.reshape(1, Y.values.shape[0]).ravel(), labels))
    #print(metrics.v_measure_score(y.values.ravel(), labels))
    print(purity_score(y.values.ravel(), labels))
    #print(homogeneity_score(y.values.ravel(), labels))
    print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(y.values.ravel(), labels))

def ddgraph(X, y):
    kmeans = KMeans(n_clusters = 2)
    y_kmeans = kmeans.fit_predict(X.values)
    print("SSE: ", kmeans.inertia_)
    X = X.values
    plt.figure(figsize=(8,6))
    #Visualising the clusters
    x0 = X[y_kmeans == 0, 0]
    x1 = X[y_kmeans == 0, 1]
    labels = kmeans.labels_
    x01 = X[y_kmeans == 1, 0]
    x11 = X[y_kmeans == 1, 1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x0, x1, .05, c = 'red', label = '0',alpha = .4, s=20)
    ax.scatter(x01, x11, -.05, c = 'blue', label = '1', alpha = .4, marker = 'x', s=20)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_zlim((-.75, .75))
    plt.show()



#pre_proc()
X = pd.read_csv("X_scaled.csv")
Y = pd.read_csv("Yx.csv")
#elbow(X)
#ddgraph(X, Y)
kmean(X,Y)

