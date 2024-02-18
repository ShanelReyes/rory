import numpy as np
import json

"""
Description: Interface used for the expected result of the liu scheme
"""
class MetricsResultInternal(object):
	def __init__(self,**kwargs):
		self.silhouette_coefficient = kwargs.get("silhouette_coefficient",0)
		self.davies_bouldin_index   = kwargs.get("davies_bouldin_index",0)
		self.calinski_harabaz_index = kwargs.get("calinski_harabaz_index",0)
		self.dunn_index             = kwargs.get("dunn_index",0)
	
	def __str__(self) -> str:
		return "{}, {}, {}, {}".format(self.silhouette_coefficient, self.davies_bouldin_index, self.calinski_harabaz_index, self.dunn_index)
	
	def toJson(self):
		return json.dumps(self.__dict__)

	def toDict(self):
		return self.__dict__
