from .LoadBalancingAlgorithm import LoadBalancingAlgorithm

class RoundRobin(LoadBalancingAlgorithm):

    def __init__(self,**kwargs):
        self.bins   = {} 
        self.total  = 0
        self.prefix = kwargs.get("prefix","scw-")
        self.n = kwargs.get("n",1)
        if self.n == 0:
            self.n=1
        for i in range(self.n):
            binId = "{}{}".format(self.prefix,i)
            self.bins[binId] = 0
    
    def getTotalRequests(self):
        total = 0 
        for (key,balls) in self.bins.items():
            total+= len(balls)
        return total

    def add_bin(self,**kws):
        binId = kws.get("binId")
        balls = kws.get("balls",0)
        self.bins[binId] = 0

    def balance(self):
        binIndex   = self.total  % self.n
        binId      = "{}{}".format(self.prefix,binIndex)
        self.bins[binId] = self.bins[binId]+1
        self.total += 1
        return binId
        
        




