from .LoadBalancingAlgorithm import LoadBalancingAlgorithm
import numpy as np

class TwoChoices(LoadBalancingAlgorithm):
    def __init__(self,**kwargs):
        self.bins   = {} 
        self.total  = 0
        self.prefix = kwargs.get("prefix","scw-")
        self.n      = kwargs.get("n",1)
        for i in range(self.n):
            binId = "{}{}".format(self.prefix,i)
            self.bins[binId] = 0
    
    def balance(self):
        x = np.random.randint(0,self.n)
        xId = "{}{}".format(self.prefix,x)
        y = np.random.randint(0,self.n)
        yId = "{}{}".format(self.prefix,y)
        self.total+= 1 
        
        if(self.bins[xId] < self.bins[yId]):
            self.bins[xId] += 1
            return xId
        else:
            self.bins[yId] += 1
            return yId