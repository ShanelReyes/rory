import numpy as np

"""
Description: Interface used for the expected result of the liu scheme
"""
class MetricsResultExternal(object):
	def __init__(self,**kwargs):
		self.adjusted_mutual_information = kwargs.get("adjusted_mutual_information",0)
		self.fowlkes_mallows_index       = kwargs.get("fowlkes_mallows_index",0)
		self.adjusted_rand_index         = kwargs.get("adjusted_rand_index",0)
		self.jaccard_index               = kwargs.get("jaccard_index",0)

	def __str__(self) -> str:
		return "{}, {}, {}, {}".format(self.adjusted_mutual_information, self.fowlkes_mallows_index, self.adjusted_rand_index, self.jaccard_index)
