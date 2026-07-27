import json


class MetricsResultInternal:
	"""Interface for internal clustering validation metrics.

	Holds scores that evaluate clustering quality based solely on the data
	and the computed labels (no ground truth required), including the
	Silhouette coefficient, Davies-Bouldin index, Calinski-Harabasz index,
	and Dunn index.

	Keyword Args:
		silhouette_coefficient (float, optional): Silhouette score. Defaults
			to 0.
		davies_bouldin_index (float, optional): Davies-Bouldin score.
			Defaults to 0.
		calinski_harabaz_index (float, optional): Calinski-Harabasz score.
			Defaults to 0.
		dunn_index (float, optional): Dunn index. Defaults to 0.
	"""
	def __init__(self,**kwargs):
		self.silhouette_coefficient = kwargs.get("silhouette_coefficient",0)
		self.davies_bouldin_index   = kwargs.get("davies_bouldin_index",0)
		self.calinski_harabaz_index = kwargs.get("calinski_harabaz_index",0)
		self.dunn_index             = kwargs.get("dunn_index",0)

	def __str__(self) -> str:
		"""Returns a CSV-formatted string of all internal metrics.

		Returns:
			str: Comma-separated values of the four internal indices.
		"""
		return "{}, {}, {}, {}".format(self.silhouette_coefficient, self.davies_bouldin_index, self.calinski_harabaz_index, self.dunn_index)

	def toJson(self):
		"""Serializes the metrics result to a JSON string.

		Returns:
			str: JSON representation of the internal metrics.
		"""
		return json.dumps(self.__dict__)

	def toDict(self):
		"""Returns the metrics result as a plain dictionary.

		Returns:
			dict: Dictionary of all internal metric values.
		"""
		return self.__dict__
