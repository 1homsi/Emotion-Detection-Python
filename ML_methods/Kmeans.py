from sklearn.datasets import make_blobs
import pandas as pd
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from array import array
from numpy import *

def kmeans_visualizer():
       dataset, classes = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=0.5, random_state=0)
       # make as panda dataframe for easy understanding
       df = pd.DataFrame(dataset, columns=['var1', 'var2'])
       df.head(2)
       model = KMeans()
       visualizer = KElbowVisualizer(model, k=(1,12)).fit(df)
       visualizer.show()
       kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0).fit(df)
       kmeans.labels_  # same as kmeans.predict(df)
       array([2, 0, 3, 1, 2, 3, 0, 3, 3, 3, 3, 2, 0, 0, 2, 3, 1, 1, 1, 2, 1, 0,
              2, 0, 2, 2, 1, 2, 2, 3, 1, 3, 0, 2, 0, 3, 0, 3, 3, 1, 1, 1, 1, 3,
              2, 0, 3, 1, 1, 3, 1, 0, 0, 1, 3, 1, 0, 2, 3, 2, 1, 3, 2, 3, 1, 3,
              2, 1, 0, 0, 2, 2, 3, 3, 0, 1, 0, 0, 2, 2, 1, 3, 2, 0, 0, 3, 3, 2,
              0, 0, 1, 1, 1, 3, 3, 2, 0, 1, 3, 3, 1, 2, 2, 1, 1, 0, 3, 2, 2, 3,
              1, 0, 0, 2, 2, 3, 0, 0, 1, 3, 1, 0, 3, 2, 3, 0, 3, 0, 2, 3, 0, 2,
              0, 1, 1, 0, 1, 1, 2, 1, 2, 0, 2, 2, 0, 2, 3, 2, 0, 1, 1, 1, 3, 0,
              2, 3, 1, 0, 1, 2, 1, 2, 2, 0, 0, 1, 3, 2, 2, 0, 2, 3, 0, 1, 1, 1,
              3, 3, 0, 3, 0, 2, 3, 2, 3, 0, 0, 1, 3, 1, 2, 2, 3, 1, 0, 0, 0, 3,
              1, 2], dtype=int32)

       kmeans.inertia_
       94.02242630751765

       kmeans.n_iter_
       2

       kmeans.cluster_centers_
       array([[-1.60782913,  2.9162828 ],
              [-1.33173192,  7.7400479 ],
              [ 2.06911036,  0.96146833],
              [ 0.91932803,  4.34824615]])

       Counter(kmeans.labels_)
       # output
       Counter({2: 50, 0: 50, 3: 50, 1: 50})

       sns.scatterplot(data=df, x="var1", y="var2", hue=kmeans.labels_)
       plt.show()

       sns.scatterplot(data=df, x="var1", y="var2", hue=kmeans.labels_)
       plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
                   marker="X", c="r", s=80, label="centroids")
       plt.legend()
       plt.show()
