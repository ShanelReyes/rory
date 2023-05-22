from .LoadBalancingAlgorithm import LoadBalancingAlgorithm
import numpy as np

class Random(LoadBalancingAlgorithm):
    def __init__(self,**kwargs):
        self.bins   = {} 
        self.total  = 0
        self.prefix = kwargs.get("prefix","scw-")
        self.n      = kwargs.get("n",1)
        for i in range(self.n):
            binId = "{}{}".format(self.prefix,i)
            self.bins[binId] = 0
    
    def balance(self):
        x              = np.random.randint(0,self.n)
        xId            = "{}{}".format(self.prefix,x)
        self.total     += 1
        self.bins[xId] += 1 
        return xId