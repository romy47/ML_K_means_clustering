import matplotlib.pyplot as plot
from matplotlib import style
style.use('ggplot')
from sklearn.cluster import KMeans

data = [[1,2],
        [2,6],
        [3,2],
        [10,7],
        [8,7],
        [9,5]]

clf = KMeans(n_clusters=2)
clf.fit(data)

centroids = clf.cluster_centers_
labels = clf.labels_
colors = 10*['g.', 'b.', 'r.']

for i in range(len(data)):
    plot.plot(data[i][0], data[i][1], colors[labels[i]], markersize = 25)
plot.scatter(centroids[:,0], centroids[:,1], marker='x', s=150, linewidths=5)
plot.show()
