from abc import ABC,abstractmethod

"""
Description: Describes a load balancing algorithm
"""
class LoadBalancingAlgorithm(ABC):
    @abstractmethod
    def balance(self):
        pass