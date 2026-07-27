

class MetricsResultExternal:
	"""Interface for external clustering validation metrics.

	Holds scores that compare the predicted clustering against a known
	ground-truth labeling, including Adjusted Mutual Information, Fowlkes-
	Mallows, Adjusted Rand, and Jaccard indices.

	Keyword Args:
		adjusted_mutual_information (float, optional): Adjusted Mutual
			Information score. Defaults to 0.
		fowlkes_mallows_index (float, optional): Fowlkes-Mallows score.
			Defaults to 0.
		adjusted_rand_index (float, optional): Adjusted Rand index.
			Defaults to 0.
		jaccard_index (float, optional): Jaccard coefficient. Defaults
			to 0.
	"""
	def __init__(self,**kwargs):
		self.adjusted_mutual_information = kwargs.get("adjusted_mutual_information",0)
		self.fowlkes_mallows_index       = kwargs.get("fowlkes_mallows_index",0)
		self.adjusted_rand_index         = kwargs.get("adjusted_rand_index",0)
		self.jaccard_index               = kwargs.get("jaccard_index",0)

	def __str__(self) -> str:
		"""Returns a CSV-formatted string of all external metrics.

		Returns:
			str: Comma-separated values of the four external indices.
		"""
		return "{}, {}, {}, {}".format(self.adjusted_mutual_information, self.fowlkes_mallows_index, self.adjusted_rand_index, self.jaccard_index)
