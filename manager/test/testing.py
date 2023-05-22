#  ______________________________
import os
import sys
from pathlib import Path
path_root      = Path(__file__).parent.absolute()
(path_root, _) = os.path.split(path_root)
sys.path.append(str(path_root))
# _________________________________
import unittest
from src.load_balancing.round_robin import RoundRobin
from src.load_balancing.random import Random
from src.load_balancing.two_choices import TwoChoices

# from src.loadBalancing.ClusteringRequest import ClusteringRequest

class Test(unittest.TestCase):

    def test_round_robin(self):
        rb = RoundRobin( n = 5)
        tc = TwoChoices(n = 5)
        rnd=  Random(n = 5)
        for i in range(10):
            binId = rb.balance()
            print("RB - BALANCE[{}] {}".format(i,binId))
            binId = tc.balance()
            print("2c - BALANCE[{}] {}".format(i,binId))
            binId = rnd.balance()
            print("RND - BALANCE[{}] {}".format(i,binId))
            print("_"*40)
        print(rb.bins)
        print(tc.bins)
        print(rnd.bins)

if __name__ == "__main__":
    unittest.main()