"""
Test Apocalipse environment
"""
import os
import shutil
import pytest
import mock
from hypothesis import given, example, settings, strategies as st
from hypothesis.extra.numpy import arrays, array_shapes
from cryptotrader.envs.trading import TradingEnvironment, PaperTradingEnvironment, BacktestDataFeed
from cryptotrader.utils import convert_to, array_normalize, array_softmax, floor_datetime
from cryptotrader.spaces import Box, Tuple
import numpy as np
import pandas as pd
from decimal import Decimal
from cryptotrader.exchange_api.poloniex import Poloniex
from datetime import datetime, timezone

from .mocks import *

# Fixtures
@pytest.fixture
def fresh_env():
    yield TradingEnvironment(period=5, obs_steps=30, tapi=tapi, fiat="USDT", name='env_test')
    shutil.rmtree(os.path.join(os.path.abspath(os.path.curdir), 'logs'))

@pytest.fixture
def ready_env():
    with mock.patch('cryptotrader.envs.trading.datetime') as mock_datetime:
        # mock_datetime.now.return_value = datetime.fromtimestamp(1507990500.000000).astimezone(timezone.utc)
        mock_datetime.now.return_value = datetime.fromtimestamp(np.choose(np.random.randint(low=10, high=len(indexes)),
                                                                          indexes)).astimezone(timezone.utc)
        mock_datetime.fromtimestamp = lambda *args, **kw: datetime.fromtimestamp(*args, **kw)
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

        env = PaperTradingEnvironment(period=5, obs_steps=10, tapi=tapi, fiat="USDT", name='env_test')
        # env.add_pairs("USDT_BTC", "USDT_ETH")
        # env.fiat = "USDT"
        env.balance = env.get_balance()
        env.crypto = {"BTC": Decimal('1.00000000'), 'ETH': Decimal('0.50000000')}
        yield env
        shutil.rmtree(os.path.join(os.path.abspath(os.path.curdir), 'logs'))

@pytest.fixture
def data_feed():
    df = BacktestDataFeed(tapi, period=5, pairs=["USDT_BTC", "USDT_ETH"], balance={"BTC":'1.00000000',
                                                                                    "ETH":'0.50000000',
                                                                                    "USDT":'100.00000000'})
    yield df

# DATA FEED TESTS
def test_returnBalances(data_feed):
    balance = data_feed.returnBalances()
    assert isinstance(balance, dict)
    data_feed.balance = {"BTC": '10.00000000'}
    balance = data_feed.returnBalances()
    assert isinstance(balance, dict)
    assert balance["BTC"] == '10.00000000'
    with pytest.raises(AssertionError):
        data_feed.balance = 10

def test_returnFeeInfo(data_feed):
    fee = data_feed.returnFeeInfo()
    assert isinstance(fee, dict)
    assert fee['makerFee'] == '0.00150000'

# BACKTEST AND PAPERTRAING ENVIRONMENT TESTS
def test_env_name(fresh_env):
    assert fresh_env.name == 'env_test'

class Test_env_setup(object):
    @classmethod
    def setup_class(cls):
        cls.env = TradingEnvironment(period=5, obs_steps=10, tapi=tapi, fiat="USDT", name='env_test')

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(os.path.join(os.path.abspath(os.path.curdir), 'logs'))

    def test_reset_status(self):
        self.env.reset_status()
        assert self.env.status == {'OOD': False, 'Error': False, 'ValueError': False, 'ActionError': False, "NotEnoughFiat": False}

    def test_add_pairs(self):
        self.env.add_pairs("USDT_BTC")
        assert "USDT_BTC" in self.env.pairs

    @given(value=st.integers(max_value=1000))
    def test_obs_steps(self, value):
        if value >= 3:
            self.env.obs_steps = value
            assert self.env.obs_steps == value
        else:
            with pytest.raises(AssertionError):
                self.env.obs_steps = value

    @given(value=st.integers(max_value=1000))
    def test_period(self, value):
        if value >= 1:
            self.env.period = value
            assert self.env.period == value
        else:
            with pytest.raises(AssertionError):
                self.env.period = value

def test_get_ohlc(ready_env):
    env = ready_env
    for data in tapi.returnChartData()[:-env.obs_steps]:
        for pair in env.pairs:
            df = env.get_ohlc(pair, index=pd.date_range(end=data['date'], freq="%dT" % env.period, periods=env.obs_steps))
            assert isinstance(df, pd.DataFrame)
            assert df.shape[0] == env.obs_steps
            assert list(df.columns) == ['open','high','low','close','volume']
            assert df.index.freqstr == '%dT' % env.period

def test_get_history(ready_env):
    env = ready_env
    df = env.get_history()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == env.obs_steps
    assert set(df.columns.levels[0]) == set(env.pairs)
    assert list(df.columns.levels[1]) == ['open', 'high', 'low', 'close', 'volume']
    assert df.index.freqstr == '%dT' % env.period
    assert type(df.values.all()) == Decimal

    for data in tapi.returnChartData()[-env.obs_steps:]:
        df = env.get_history(end=datetime.fromtimestamp(data['date']))
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == env.obs_steps
        assert set(df.columns.levels[0]) == set(env.pairs)
        assert list(df.columns.levels[1]) == ['open', 'high', 'low', 'close', 'volume']
        assert df.index.freqstr == '%dT' % env.period
        assert type(df.values.all()) == Decimal

def test_get_balance(ready_env):
    env = ready_env
    balance = env.get_balance()

    portfolio = []
    for pair in env.symbols:
        symbol = pair.split('_')
        for s in symbol:
            portfolio.append(s)

    portfolio = set(portfolio)

    assert set(balance.keys()).issubset(portfolio)

def test_fiat(fresh_env):
    env = fresh_env

    # with pytest.raises(AssertionError):
    #     env.fiat = "USDT"

    # env.add_pairs("USDT_BTC")
    env.fiat = "USDT"
    # assert env.fiat == Decimal('0.00000000')

    # with pytest.raises(KeyError):
    #     env.fiat

    env.fiat = 0
    assert env.fiat == Decimal('0.00000000')

    for i in range(10):
        env.fiat = i
        assert env.fiat == Decimal(i)

        value = np.random.rand() * i
        env.fiat = value
        assert env.fiat == convert_to.decimal(value)

    timestamp = env.timestamp
    env.fiat = {"USDT": 10, 'timestamp': timestamp}
    assert env.portfolio_df.get_value(timestamp, "USDT") == 10

def test_crypto(fresh_env):
    env = fresh_env
    env.add_pairs("USDT_BTC")
    env.fiat = "USDT"

    with pytest.raises(AssertionError):
        env.crypto = []
        env.crypto = 0
        env.crypto = '0'

    balance = env.get_balance()
    env.crypto = balance
    for symbol, value in env.crypto.items():
        assert value == convert_to.decimal(balance[symbol])
        assert symbol in env.symbols
        assert env._fiat not in env.crypto

    timestamp = env.timestamp
    env.crypto = {"BTC": 10, 'timestamp': timestamp}
    assert env.portfolio_df.get_value(timestamp, "BTC") == Decimal('10')

def test_balance(fresh_env):
    env = fresh_env
    env.add_pairs("USDT_BTC", "USDT_ETH")
    env.fiat = "USDT"

    with pytest.raises(AssertionError):
        env.balance = []
        env.balance = 0
        env.balance = '0'

    env.balance = env.get_balance()
    for symbol, value in env.balance.items():
        assert value == convert_to.decimal(env.balance[symbol])
        assert symbol in env.symbols
        assert env._fiat not in env.crypto.keys()

def test_get_close_price(ready_env):
    env = ready_env
    env.obs_df = env.get_history()

    price = env.get_open_price("BTC")
    assert isinstance(price, Decimal)
    assert price == env.obs_df["USDT_BTC"].open.iloc[-1]

    for i in env.obs_df.index:
        price = env.get_open_price("BTC", i)
        assert isinstance(price, Decimal)
        assert price == env.obs_df["USDT_BTC"].open.loc[i]

def test_get_fee(ready_env):
    env = ready_env
    fee = env.get_fee("BTC")
    assert isinstance(fee, Decimal)
    assert fee == Decimal('0.00250000')

    fee = env.get_fee("BTC", "makerFee")
    assert isinstance(fee, Decimal)
    assert fee == Decimal('0.00150000')

    with pytest.raises(AssertionError):
        fee = env.get_fee("BTC", 'wrong_str')

def test_calc_total_portval(ready_env):
    env = ready_env
    env.obs_df = env.get_history()
    portval = env.calc_total_portval()
    assert isinstance(portval, Decimal)
    assert portval >= Decimal('0.00000000')

def test_calc_posit(ready_env):
    env = ready_env
    env.obs_df = env.get_history()
    total_posit = Decimal('0E-8')
    portval = env.calc_total_portval()
    for symbol in env.symbols:
        posit = env.calc_posit(symbol, portval)
        assert isinstance(posit, Decimal)
        assert Decimal('0.00000000') <= posit <= Decimal('1.00000000')
        total_posit += posit
    assert total_posit - Decimal('1.00000000') <= Decimal('1E-8')

def test_get_previous_portval(ready_env):
    env = ready_env
    env.obs_df = env.get_history()
    with pytest.raises(KeyError):
        portval = env.get_last_portval()

    env.portval = 10
    portval = env.get_last_portval()
    assert portval == Decimal('10')

def test_get_sampled_portfolio(ready_env):
    # TODO: WRITE TEST
    env = ready_env
    env.reset()

    assert env.get_sampled_portfolio().shape == (1, 4)

def test_get_reward(ready_env):
    env = ready_env
    env.reset()
    r = env.get_reward()
    assert isinstance(r, float)
    # assert r == float()
    env.fiat = Decimal(1)
    a = np.zeros(len(env.pairs) + 1)
    a[-1] = 1
    env.step(a)
    n_tests = 100
    for i, j in zip(np.random.random(n_tests), np.random.random(n_tests)):
        # env.reset()

        # env.step(a)
        env.fiat = Decimal(i)
        # env.portval = env.calc_total_portval()
        # env.fiat = Decimal(j)
        r = env.get_reward()
        # assert r - Decimal(j / i) < Decimal("1e-4"), r - Decimal(j / i)
        assert np.allclose(r, np.log(j) - np.log(i)), (r, np.log(j) - np.log(i))


index = np.choose(np.random.randint(low=10, high=len(indexes)), indexes)
class Test_env_reset(object):
    @classmethod
    def setup_class(cls):
        with mock.patch('cryptotrader.envs.trading.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.fromtimestamp(index).astimezone(timezone.utc)
            mock_datetime.fromtimestamp = lambda *args, **kw: datetime.fromtimestamp(*args, **kw)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            cls.env = PaperTradingEnvironment(period=5, obs_steps=10, tapi=tapi, fiat="USDT", name='env_test')
            # cls.env.add_pairs("USDT_BTC", "USDT_ETH")
            # cls.env.fiat = "USDT"

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(os.path.join(os.path.abspath(os.path.curdir), 'logs'))

    @mock.patch.object(PaperTradingEnvironment, 'timestamp',
                       floor_datetime(datetime.fromtimestamp(index).astimezone(timezone.utc), 5))
    def test_reset(self):
        obs = self.env.reset()

        # Assert observation
        assert isinstance(self.env.obs_df, pd.DataFrame) and self.env.obs_df.shape[0] == self.env.obs_steps
        assert isinstance(obs, pd.DataFrame) and obs.shape[0] == self.env.obs_steps
        # Assert taxes
        assert list(self.env.tax.keys()) == self.env.symbols
        # Assert portfolio log
        assert isinstance(self.env.portfolio_df, pd.DataFrame) and self.env.portfolio_df.shape[0] == 1
        assert list(self.env.portfolio_df.columns) == list(self.env.symbols) + ['portval']
        # Assert action log
        assert isinstance(self.env.action_df, pd.DataFrame) and self.env.action_df.shape[0] == 1
        assert list(self.env.action_df.columns) == list(self.env.symbols) + ['online']
        # Assert balance
        assert list(self.env.balance.keys()) == list(self.env.symbols)
        for symbol in self.env.balance:
            assert isinstance(self.env.balance[symbol], Decimal)

@pytest.mark.incremental
class Test_env_step(object):
    # TODO: CHECK THIS TEST
    @classmethod
    def setup_class(cls):
        with mock.patch('cryptotrader.envs.trading.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.fromtimestamp(index).astimezone(timezone.utc)
            mock_datetime.fromtimestamp = lambda *args, **kw: datetime.fromtimestamp(*args, **kw)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            cls.env = PaperTradingEnvironment(period=5, obs_steps=10, tapi=tapi, fiat="USDT", name='env_test')
            # cls.env.add_pairs("USDT_BTC", "USDT_ETH")
            # cls.env.fiat = "USDT"
            cls.env.reset()
            cls.env.fiat = 100
            cls.env.reset_status()

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(os.path.join(os.path.abspath(os.path.curdir), 'logs'))

    @given(arrays(dtype=np.float32,
                  shape=(3,),
                  elements=st.floats(allow_nan=False, allow_infinity=False, max_value=1e8, min_value=0)))
    @settings(max_examples=50)
    def test_simulate_trade(self, action):
        # Normalize action vector
        action = array_normalize(action, False)

        assert action.sum() - Decimal('1.00000000') < Decimal('1E-8'), action.sum() - Decimal('1.00000000')

        # Get timestamp
        timestamp = self.env.obs_df.index[-1]
        # Call method
        self.env.simulate_trade(action, timestamp)
        # Assert position
        for i, symbol in enumerate(self.env.symbols):
            assert self.env.action_df.get_value(timestamp, symbol) - convert_to.decimal(action[i]) <= Decimal('1E-8')
        # Assert amount
        for i, symbol in enumerate(self.env.symbols):
            if symbol not in self.env._fiat:
                assert self.env.portfolio_df.get_value(self.env.portfolio_df[symbol].last_valid_index(), symbol) - \
                       self.env.action_df.get_value(timestamp, symbol) * self.env.calc_total_portval(timestamp) / \
                        self.env.get_open_price(symbol, timestamp) <= convert_to.decimal('1E-4')

    @mock.patch.object(PaperTradingEnvironment, 'timestamp',
                       floor_datetime(datetime.fromtimestamp(index).astimezone(timezone.utc), 5))
    @given(arrays(dtype=np.float32,
                  shape=(3,),
                  elements=st.floats(allow_nan=False, allow_infinity=False, max_value=1e8, min_value=0)))
    @settings(max_examples=50)
    def test_step(self, action):
        # obs = self.env.reset()
        action = array_softmax(action)
        obs, reward, done, status = self.env.step(action)

        # Assert returned obs
        assert isinstance(obs, pd.DataFrame)
        assert obs.shape[0] == self.env.obs_steps
        assert set(obs.columns.levels[0]) == set(list(self.env.pairs) + [self.env._fiat])

        # Assert reward
        assert isinstance(reward, np.float64)
        assert reward not in (np.nan, np.inf)

        # Assert done
        assert isinstance(done, bool)

        # Assert status
        assert status == self.env.status
        for key in status:
            assert status[key] == False

# LIVETRADING ENVIRONMENT TESTS


if __name__ == '__main__':
    pytest.main()


