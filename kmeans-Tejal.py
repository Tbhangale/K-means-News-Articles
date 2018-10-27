from functools import partial
from itertools import count
import pandas as pd
import math
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score
np.seterr(divide='ignore', invalid='ignore')

data = pd.read_csv('D:\Sem 2\Temporal and spatial data\HW3\sunday\Alldata.csv', index_col=0)
labels = data['Categories']
label_to_number = defaultdict(partial(next, count(1)))
label_num = [label_to_number[label] for label in labels]
train_X = data.drop('Categories', axis=1)
train_X_np = np.array(train_X)

class KMeans(object):
    def __init__(self, n_clusters=20, max_iter=1, random_state=321, dist = 'euclidean'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.dist = dist

    def fit(self, X):
        if self.random_state:
            np.random.seed(self.random_state)
        initial = np.random.permutation(X.shape[0])[:self.n_clusters]
        self.cluster_centers_ = X[initial]

        for _ in range(self.max_iter):
            self.labels_ = [self._nearest(self.cluster_centers_, x) for x in X]
            indices = [[i for i, l in enumerate(self.labels_) if l == j]
                        for j in range(self.n_clusters)]
            X_by_cluster = [X[i] for i in indices]
            # update the clusters
            self.cluster_centers_ = [c.sum(axis=0) / len(c) for c in X_by_cluster]
        # sum of square distances from the closest cluster
        self.inertia_ = sum(((self.cluster_centers_[l] - x)**2).sum()
                            for x, l in zip(X, self.labels_))
        return self

    def _nearest(self, clusters, x):
        return np.argmin([self._distance(x, c) for c in clusters])
                           
    def predict(self, X):
        return self.labels_

    def transform(self, X):
        return [[self._distance(x, c) for c in self.cluster_centers_] for x in X]

    def fit_predict(self, X):
        return self.fit(X).predict(X)

    def centers(self):
        return self.cluster_centers_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def score(self, X):
        return self.inertia_

    def _distance(self, x, y):
        if self.dist == 'jaccard':
            intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
            union_cardinality = len(set.union(*[set(x), set(y)]))
            j_sim = intersection_cardinality / float(union_cardinality)
            return 1 - j_sim

        elif self.dist == 'euclidean':
            sumOfSquares = 0
            for i in range(1, len(x)):
                sumOfSquares += ((x[i] - y[i]) ** 2)
            sumOfSquares = float(math.sqrt(sumOfSquares))
            return sumOfSquares

        elif self.dist == 'cosine':
            dot_product = np.dot(x, y)
            norm_x = np.linalg.norm(x)
            norm_y = np.linalg.norm(y)
            c_sim = dot_product / (norm_x * norm_y)
            return 1 - c_sim

all_sse = []
acc_score = {}
sse_dict = {}
distances = ['euclidean', 'jaccard', 'cosine']
n_clusters = 20
for d in distances:
    labels = np.array(KMeans(dist=d).fit_predict(train_X_np))
    centers = KMeans().fit(train_X_np).centers()
    train_X[d+'_label'] = labels
    for p in range(n_clusters):
        center = centers[p]
        clust_data = train_X_np[labels==p]
        all_sse=[]
        sse = 0
        for dp in clust_data:
            sse += ((center - dp)**2).sum()
            all_sse.append(sse)
    sse_dict[d] = np.mean(all_sse)
    acc_score[d] = accuracy_score(label_num, labels)

pd.DataFrame(sse_dict, index=['SSE']).to_csv('error_profiles_alldata.csv')
train_X.to_csv('predicted_clusters_alldata.csv')
pd.DataFrame(acc_score, index=['Accuracy Score']).to_csv('accuracy_scores_alldata.csv')
