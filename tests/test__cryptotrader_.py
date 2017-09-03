import unittest

import mock

from cryptotrader.agents import apriori


class TestDummyTrader(unittest.TestCase):

    def setUp(self):
        self.env = mock.Mock()
        self.agent = apriori.DummyTrader(self.env)

    def test_act(self):
        pass


if __name__ == '__main__':
    unittest.main()