import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def delta(ck, cl):
	"""Computes the minimum pairwise distance between two clusters.

	Args:
		ck (np.ndarray): Points in the first cluster, shape (n_k, p).
		cl (np.ndarray): Points in the second cluster, shape (n_l, p).

	Returns:
		float: The smallest Euclidean distance between any point in ck
			and any point in cl.
	"""
	values = np.ones([len(ck), len(cl)])*10000

	for i in range(0, len(ck)):
		for j in range(0, len(cl)):
			values[i, j] = np.linalg.norm(ck[i]-cl[j])

	return np.min(values)

def big_delta(ci):
	"""Computes the maximum intra-cluster distance (diameter).

	Args:
		ci (np.ndarray): Points in the cluster, shape (n_i, p).

	Returns:
		float: The largest pairwise Euclidean distance within the cluster.
	"""
	values = np.zeros([len(ci), len(ci)])

	for i in range(0, len(ci)):
		for j in range(0, len(ci)):
			values[i, j] = np.linalg.norm(ci[i]-ci[j])

	return np.max(values)

def dunn(k_list):
	"""Dunn index [CVI] for clustering validation.

	Args:
		k_list (list of np.ndarray): A list containing a numpy array for
			each cluster. Each c[K] has shape (N, p), where N is the
			number of samples and p is the feature dimension.

	Returns:
		float: The Dunn index (min inter-cluster distance / max
			intra-cluster diameter).
	"""
	deltas = np.ones([len(k_list), len(k_list)])*1000000
	big_deltas = np.zeros([len(k_list), 1])
	l_range = list(range(0, len(k_list)))

	for k in l_range:
		for l in (l_range[0:k]+l_range[k+1:]):
			deltas[k, l] = delta(k_list[k], k_list[l])

		big_deltas[k] = big_delta(k_list[k])

	di = np.min(deltas)/np.max(big_deltas)
	return di


def delta_fast(ck, cl, distances):
	"""Computes the minimum inter-cluster distance using a precomputed
	distance matrix.

	Args:
		ck (np.ndarray): Boolean mask for the first cluster.
		cl (np.ndarray): Boolean mask for the second cluster.
		distances (np.ndarray): Precomputed pairwise distance matrix.

	Returns:
		float: The minimum distance between the two clusters.
	"""
	values = distances[np.where(ck)][:, np.where(cl)]
	values = values[np.nonzero(values)]

	return np.min(values)

def big_delta_fast(ci, distances):
	"""Computes the maximum intra-cluster distance using a precomputed
	distance matrix.

	Args:
		ci (np.ndarray): Boolean mask for the cluster.
		distances (np.ndarray): Precomputed pairwise distance matrix.

	Returns:
		float: The maximum distance within the cluster.
	"""
	values = distances[np.where(ci)][:, np.where(ci)]

	return np.max(values)


def dunn_fast(points, labels):
	"""Dunn index - FAST variant using sklearn's euclidean_distances.

	Args:
		points (np.ndarray): Array of shape (N, p) of all data points.
		labels (np.ndarray): Array of shape (N,) of cluster labels.

	Returns:
		float: The Dunn index.
	"""
	distances = euclidean_distances(points)
	ks = np.sort(np.unique(labels))

	deltas = np.ones([len(ks), len(ks)])*1000000
	big_deltas = np.zeros([len(ks), 1])

	l_range = list(range(0, len(ks)))

	for k in l_range:
		for l in (l_range[0:k]+l_range[k+1:]):
			deltas[k, l] = delta_fast((labels == ks[k]), (labels == ks[l]), distances)

		big_deltas[k] = big_delta_fast((labels == ks[k]), distances)

	di = np.min(deltas)/np.max(big_deltas)
	return di


def big_s(x, center):
	"""Computes the average distance of points in a cluster to its center.

	Args:
		x (np.ndarray): Points in the cluster, shape (n, p).
		center (np.ndarray): The cluster centroid, shape (p,).

	Returns:
		float: The mean Euclidean distance from the cluster center.
	"""
	len_x = len(x)
	total = 0

	for i in range(len_x):
		total += np.linalg.norm(x[i]-center)    

	return total/len_x


def davisbouldin(k_list, k_centers):
	"""Davis-Bouldin index for clustering validation.

	Args:
		k_list (list of np.ndarray): A list containing a numpy array for
			each cluster. Each c[K] has shape (N, p).
		k_centers (np.ndarray): Array of cluster centers with shape
			(K, p).

	Returns:
		float: The Davis-Bouldin index.
	"""
	len_k_list = len(k_list)
	big_ss = np.zeros([len_k_list], dtype=np.float64)
	d_eucs = np.zeros([len_k_list, len_k_list], dtype=np.float64)
	db = 0    

	for k in range(len_k_list):
		big_ss[k] = big_s(k_list[k], k_centers[k])

	for k in range(len_k_list):
		for l in range(0, len_k_list):
			d_eucs[k, l] = np.linalg.norm(k_centers[k]-k_centers[l])

	for k in range(len_k_list):
		values = np.zeros([len_k_list-1], dtype=np.float64)
		for l in range(0, k):
			values[l] = (big_ss[k] + big_ss[l])/d_eucs[k, l]
		for l in range(k+1, len_k_list):
			values[l-1] = (big_ss[k] + big_ss[l])/d_eucs[k, l]

		db += np.max(values)
	res = db/len_k_list
	return res