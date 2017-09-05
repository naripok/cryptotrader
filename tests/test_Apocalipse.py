"""
Test Apocalipse environment
"""
import os
import shutil
import pytest
import mock
from hypothesis import given, example, settings, strategies as st
from hypothesis.extra.numpy import arrays, array_shapes

from cryptotrader.envs.driver import Apocalipse
from cryptotrader.envs.utils import SinusoidalProcess, sample_trades
from cryptotrader.utils import convert_to, array_normalize, array_softmax
from cryptotrader.spaces import Box, Tuple
import numpy as np
import pandas as pd
from decimal import Decimal

@pytest.fixture
def fresh_env():
    yield Apocalipse(name='env_test')
    shutil.rmtree(os.path.join(os.path.abspath(os.path.curdir), 'logs'))

@pytest.fixture
def keys():
    return ['btcusd', 'ltcusd', 'xrpusd', 'ethusd', 'etcusd', 'xmrusd', 'zecusd', 'iotusd', 'bchusd', 'dshusd', 'stcusd']

@pytest.fixture
def dfs(n_assets=4, freq=5):
    data_process = SinusoidalProcess(100, 2, 1000)
    dfs = []
    index = pd.DatetimeIndex(start='2017-01-01 00:00:00', end='2017-04-30 00:00:00', freq='1min')[-1000:]
    for i in range(n_assets):
        data = np.clip(data_process.sample_block(), a_min=1e-12, a_max=np.inf)
        dfs.append(sample_trades(pd.DataFrame(data, columns=['trade_px', 'trade_volume'], index=index), freq=str(freq)+'min'))

    return dfs

# Tests
def test_env_name(fresh_env):
    assert fresh_env.name == 'env_test'

@given(freq=st.one_of(st.integers(), st.floats()))
def test_set_freq(fresh_env, freq):
    if isinstance(freq, int) and freq >= 1:
        fresh_env.set_freq(freq)
        assert fresh_env.freq == freq
    else:
        with pytest.raises(AssertionError):
            fresh_env.set_freq(freq)

@given(obs_steps=st.one_of(st.integers(), st.floats()))
def test_set_obs_steps(fresh_env, obs_steps):
    if isinstance(obs_steps, int) and obs_steps >= 3:
        fresh_env.set_obs_steps(obs_steps)
        assert fresh_env.obs_steps == obs_steps
        assert fresh_env.step_idx == obs_steps
        assert fresh_env.offset == obs_steps
    else:
        with pytest.raises(AssertionError):
            fresh_env.set_obs_steps(obs_steps)

def test_set_training_stage(fresh_env):
    fresh_env.set_training_stage(True)
    assert fresh_env._is_training == True
    fresh_env.set_training_stage(False)
    assert fresh_env._is_training == False

def test_set_observation_space(fresh_env):
    fresh_env.set_observation_space()
    assert isinstance(fresh_env.observation_space, Tuple)
    assert isinstance(fresh_env.observation_space.spaces, list)
    for space in fresh_env.observation_space.spaces:
        assert isinstance(space, Box)

@pytest.mark.incremental
class Test_env_setup(object):
    @classmethod
    def setup_class(cls):
        cls.env = Apocalipse(name='test_env_setup')
        cls.env.set_freq(5)
        cls.env.set_obs_steps(5)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(os.path.join(os.path.abspath(os.path.curdir), 'logs'))

    def test_add_df(self, dfs, keys):
        for i in range(len(dfs)):
            self.env.add_df(df=dfs[i], symbol=keys[i])
            assert len(self.env.dfs) == i + 1
        assert [isinstance(df, pd.DataFrame) for df in self.env.dfs.values()]
        # TODO other test cases

    def test_add_symbol(self, keys):
        for i in range(len(self.env.dfs)):
            self.env.add_symbol(symbol=keys[i])
            assert keys[i] in self.env.symbols
        # TODO other test cases

    @given(amount=st.floats(max_value=1e18, allow_infinity=False, allow_nan=False))
    def test_set_init_crypto(self, amount):
        symbols = self.env.df.columns.levels[0][:-1]
        for symbol in symbols:
            if amount >= Decimal('0.0'):
                self.env.set_init_crypto(amount, symbol)
                assert self.env.init_crypto[symbol] == convert_to.decimal(amount)
            else:
                with pytest.raises(AssertionError):
                    self.env.set_init_crypto(amount, symbol)

    @given(amount=st.floats(max_value=1e18, allow_infinity=False, allow_nan=False))
    def test_set_init_fiat(self, amount):
        if amount >= Decimal('0.0'):
            self.env.set_init_fiat(amount)
            assert self.env.init_fiat == convert_to.decimal(amount)
        else:
            with pytest.raises(AssertionError):
                self.env.set_init_fiat(amount)

    @given(tax=st.floats(max_value=1e18, allow_infinity=False, allow_nan=False))
    def test_set_tax(self, tax):
        symbols = self.env.df.columns.levels[0][:-1]
        for symbol in symbols:
            if 0.0 <= tax <= 1.0:
                self.env.set_tax(tax, symbol)
                assert self.env.optax[symbol] == convert_to.decimal(tax)
            else:
                with pytest.raises(AssertionError):
                    self.env.set_tax(tax, symbol)

    def test__reset_status(self):
        self.env._reset_status()
        assert self.env.status == {'OOD': False, 'Error': False, 'ValueError': False, 'ActionError': False,
                       'OnlineActionError': False, 'OnlineValueError': False}

    def test_clear_dfs(self):
        self.env.clear_dfs()
        assert self.env.dfs == {}

    def test_set_action_space(self):
        self.env.set_action_space()
        assert isinstance(self.env.action_space, Box)
        assert self.env.action_space.low.shape[0] == len(self.env.symbols)

    @given(init_fiat=st.floats(max_value=1e18, min_value=0.0, allow_infinity=False, allow_nan=False),
           init_crypto=st.floats(max_value=1e18, min_value=0.0, allow_infinity=False, allow_nan=False),
           )
    @settings(max_examples=10)
    def test_reset(self, init_fiat, init_crypto):
        # It not training
        for symbol in self.env.df.columns.levels[0]:
            if symbol is not 'fiat':
                self.env.set_init_crypto(init_crypto, symbol)
        self.env.set_init_fiat(init_fiat)
        self.env.set_training_stage(False)

        obs = self.env.reset(reset_funds=True, reset_global_step=True, reset_results=True)

        assert isinstance(obs, pd.DataFrame)
        assert obs.shape[0] == self.env.obs_steps
        assert self.env.global_step == 0
        assert self.env.step_idx == self.env.offset

        assert self.env.df.iloc[:self.env.step_idx].fiat.amount.values.all() == convert_to.decimal(init_fiat)
        for symbol in self.env.df.columns.levels[0]:
            if symbol is not 'fiat':
                assert symbol in self.env.symbols
                assert self.env.df[symbol].iloc[:self.env.step_idx].amount.values.all() == convert_to.decimal(init_crypto)

        # If training
        self.env.set_training_stage(True)

        obs = self.env.reset(reset_funds=True, reset_global_step=True, reset_results=True)

        assert isinstance(obs, pd.DataFrame)
        assert obs.shape[0] == self.env.obs_steps
        assert self.env.global_step == 0
        assert self.env.offset <= self.env.step_idx <= self.env.df.shape[0]

        assert self.env.df.iloc[self.env.step_idx - self.env.obs_steps:self.env.step_idx].fiat.amount.values.all() == \
               convert_to.decimal(init_fiat)
        for symbol in self.env.df.columns.levels[0]:
            if symbol is not 'fiat':
                assert symbol in self.env.symbols
                assert self.env.df[symbol].iloc[self.env.step_idx - self.env.obs_steps:self.env.step_idx].\
                           amount.values.all() == convert_to.decimal(init_crypto)
        # TODO ASSERT POSITIONS


@pytest.mark.incremental
class Test_env_step(object):
    @classmethod
    def setup_class(cls):
        cls.env = Apocalipse(name='test_env_setup')
        cls.env.set_freq(5)
        cls.env.set_obs_steps(5)
        for i in range(len(dfs())):
            cls.env.add_df(df=dfs()[i], symbol=keys()[i])
            cls.env.add_symbol(symbol=keys()[i])
            cls.env.set_init_crypto(100.0, keys()[i])
            cls.env.set_tax(0.0025, keys()[i])
        cls.env.set_init_fiat(100)

        cls.env._reset_status()
        cls.env.clear_dfs()
        cls.env.set_training_stage(True)
        cls.env.set_observation_space()
        cls.env.set_action_space()
        cls.env.reset(reset_funds=True, reset_results=True, reset_global_step=True)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(os.path.join(os.path.abspath(os.path.curdir), 'logs'))

    @given(st.floats(allow_nan=False, allow_infinity=False, max_value=1e12, min_value=-1e12))
    def test__set_posit(self, posit):
        def set_posit(posit):
            for symbol in self.env.df.columns.levels[0]:
                timestamp = self.env.df.index[self.env.step_idx]
                self.env._set_posit(posit, symbol, timestamp)
                assert self.env.posit[symbol] == convert_to.decimal(posit)
                assert self.env.df[symbol].loc[timestamp, 'position'] == convert_to.decimal(posit)
        if 0.0 <= posit <= 1.0:
            set_posit(posit)
        else:
            with pytest.raises(AssertionError):
                set_posit(posit)

    @given(st.floats(allow_nan=False, allow_infinity=False, max_value=1e12, min_value=-1e12))
    def test__set_prev_posit(self, posit):
        def set_prev_posit(posit):
            for symbol in self.env.df.columns.levels[0]:
                timestamp = self.env.df.index[self.env.step_idx]
                self.env._set_prev_posit(posit, symbol, timestamp)
                assert self.env.prev_posit[symbol] == convert_to.decimal(posit)
                assert self.env.df[symbol].loc[timestamp, 'prev_position'] == convert_to.decimal(posit)
        if 0.0 <= posit <= 1.0:
            set_prev_posit(posit)
        else:
            with pytest.raises(AssertionError):
                set_prev_posit(posit)

    def test__calc_step_portval(self):
        for symbol in self.env.df.columns.levels[0]:
            if symbol is not 'fiat':
                val = self.env.df.get_value(self.env.df.index[self.env.step_idx], (symbol, 'close')) * self.env.crypto[symbol]
                assert self.env._calc_step_portval(symbol) == val

    @given(arrays(dtype=np.float32,
                  shape=(5,),
                  elements=st.floats(allow_nan=False, allow_infinity=False, max_value=1e12, min_value=-1e12)))
    @settings(max_examples=20)
    def test__simulate_trade(self, action):
        action = array_softmax(action)
        timestamp = self.env.df.index[self.env.step_idx]
        self.env.reset(reset_funds=True, reset_results=True, reset_global_step=True)
        print("before")
        print(self.env.posit)
        self.env._simulate_trade(action, timestamp)
        print("after")
        print(action)
        print(self.env.posit)
        for i, symbol in enumerate(self.env.df.columns.levels[0]):
            assert np.allclose(np.float32(self.env.df[symbol].get_value(timestamp, 'position')), action[i], atol=5e-2), \
                (np.float32(self.env.df[symbol].get_value(timestamp, 'position')), action[i], symbol)
        # TODO PASS THIS TETS


if __name__ == '__main__':
    pytest.main()


