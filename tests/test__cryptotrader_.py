import unittest
import hypothesis

from cryptotrader.agents import apriori

class TestDummyTrader(unittest.TestCase):

    def setUp(self):
        self.agent = apriori.DummyTrader()


    def test_act(self):




if __name__ == '__main__':
    unittest.main()