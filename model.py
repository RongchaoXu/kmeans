import numpy as np
from scipy.spatial import distance


def euclDistance(vector1, vector2):
    return np.sqrt(sum(np.power(vector2 - vector1, 2)))  #求这两个矩阵的距离，vector1、2均为矩阵


def kmeans(data:np.ndarray, k=3, max_iteration=100, min_sse=0.001):
    length = data.shape[0]
    idx = np.random.randint(length, size=k)
    centers = list(data[idx, :])
    labels = [-1 for i in range(length)]
    iteration = 0
    pre_sse = 0
    while iteration < max_iteration:
        tmp = [[] for i in range(k)]
        sse = 0
        for i in range(length):
            i_label = [euclDistance(data[i], center) for center in centers]
            i_label = i_label.index(min(i_label))
            tmp[i_label].append(i)
        for i in range(k):
            cluster = [data[j] for j in tmp[i]]
            center = np.mean(cluster, axis=0)
            centers[i] = center
            for ele in cluster:
                sse += euclDistance(ele, center)
        average = sse/length
        iteration += 1
        if abs(average - pre_sse) < min_sse:
            break
        pre_sse = average
    return pre_sse


if __name__ == '__main__':
    from dataset import get_dataset
    from sklearn.cluster import KMeans
    kmeans(get_dataset('./Data for Problem 2/seeds.txt'))