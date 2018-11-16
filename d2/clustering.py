import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn import mixture
from mpl_toolkits.mplot3d import Axes3D

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 


def get_pca(X, n):
	pca = PCA(n_components=n)
	X_trs = pca.fit_transform(X)
	return X_trs

def pre_proc():
	df1 = pd.read_csv("hinselmann.csv")
	df2 = pd.read_csv("schiller.csv")
	df3 = pd.read_csv("green.csv")

	df4 = pd.concat([df1, df2, df3])
	Y = df4['consensus']

	df4.to_csv("d2_3_ds_merged.csv", index = False)
	df4 = df4.drop(['experts::0','experts::1','experts::2','experts::3','experts::4', 'experts::5', 'consensus'], axis = 1)

	X_scaled = StandardScaler().fit_transform(df4)
	X_scaled_tran = pd.DataFrame(data = X_scaled, columns = df4.columns.values).to_csv("X_scaled.csv", index = False)

	Y.to_csv("Y_to_n.csv", index = False)

def dbscan(X, Y, e):
    # Compute DBSCAN using Iris dataset
	db = DBSCAN(eps=e).fit(X)
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
				markeredgecolor='k', markersize=5)
	
		xy = X[class_member_mask & ~core_samples_mask]
		plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
				markeredgecolor='k', markersize=2)
	
	plt.title('Estimated number of clusters: %d' % n_clusters_)
	plt.show()
	print(purity_score(Y.values.reshape(1, Y.values.shape[0]).ravel(), labels))
	print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))
	print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(Y.values.reshape(1, Y.values.shape[0]).ravel(), labels))
	return metrics.adjusted_rand_score(Y.values.reshape(1, Y.values.shape[0]).ravel(), labels)
    
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
    return wcss

def ddgraph(X, y):
    kmeans = KMeans(n_clusters = 2)
    y_kmeans = kmeans.fit_predict(X)
    print("SSE: ", kmeans.inertia_)
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

def kmean(X, y):
    kmeans = KMeans(n_clusters = 2)
    y_kmeans = kmeans.fit_predict(X)
    print(y_kmeans)
    print("SSE: ", kmeans.inertia_)
    plt.figure(figsize=(8,6))
    #Visualising the clusters
    x0 = X[y_kmeans == 0, 0]
    x1 = X[y_kmeans == 0, 1]
    labels = kmeans.labels_
    x01 = X[y_kmeans == 1, 0]
    x11 = X[y_kmeans == 1, 1]
    
    plt.scatter(x0, x1, c = 'red', label = '0', s=10)
    plt.scatter(x01, x11, c = 'blue', label = '1', marker = 'x', s=10)
    
    #Plotting the centroids of the clusters
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], c = 'green', label = 'Centroids')
    plt.legend()
    #print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, y_kmeans))
    print(purity_score(y.values.reshape(1, Y.values.shape[0]).ravel(), labels))
    try:
    	print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(Y.values.reshape(1, y.values.shape[0]).ravel(), labels))
    except ValueError:
    	print("ERRO")

def plo(X, y):
    y = y.values.ravel()
    x0 = X[y == 0, 0]
    x1 = X[y == 0, 1]
    x01 = X[y == 1, 0]
    x11 = X[y == 1, 1]
    plt.figure(figsize=(8,6))
    plt.scatter(x0,x1, c = 'red', label = '0', s=10)
    plt.scatter(x01,x11, c = 'blue', label = '1', marker = 'x', s=10)
    plt.legend()

X = pd.read_csv("X_scaled.csv").values
Y = pd.read_csv("Y_to_n.csv")
#X = get_pca(X, 2)
#vec = elbow(X)
#kmean(X, Y)
#ret = dbscan(X, Y, 9.3)
#ddgraph(X, Y)
#dbscan(X,Y,5.5)
plo(X, Y)

'''
res = []
for i in np.arange(1,11,0.1):
	try:
		res.append([dbscan(X, Y, i), i])
	except ValueError:
		print("404")

res = np.array(res)

plt.plot(res[:,1], res[:,0])
plt.xlabel('Eps')
plt.ylabel('Rand score')
plt.show()
'''