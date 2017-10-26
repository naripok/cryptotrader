"""
Gym-like environment implementation for cryptocurrency trading and trade simulation

author: Fernando H'.' Canteruccio, José Olímpio Mendes
date: 17/07/2017
"""

from .. import error
from .. import seeding
from ..core import Env
from ..spaces import *
from ..utils import Logger
from .utils import *

import os
import smtplib
from datetime import datetime, timedelta
from decimal import getcontext, localcontext, ROUND_DOWN, ROUND_UP, Decimal
from time import sleep, time
import pandas as pd
import empyrical as ec
import pymongo as pm
from bitstamp.client import Trading
from bokeh.layouts import column
from bokeh.palettes import inferno
from bokeh.plotting import figure, show


# Decimal precision
getcontext().prec = 24

# Debug flag
debug = True


def make_env(test, n_assets, obs_steps=100, freq=30, tax=0.0025, init_fiat=100, init_crypto=0.0, seed=42, toy=True, files=None):
    """
    Make environment function to be called by each agent thread
    :param test:
    :param n_assets:
    :param obs_steps:
    :param freq:
    :param tax:
    :param init_fiat:
    :param init_crypto:
    :param seed:
    :param toy:
    :param files:
    :return:
    """
    # Get data
    gc.collect()
    np.random.seed(seed)

    if toy:
        dfs = make_toy_dfs(n_assets, freq)
    else:
        dfs = make_dfs(0, files, demo=True, freq=freq)

    ## ENVIRONMENT INITIALIZATION
    env = TrainingEnvironment(name='toy_env', seed=seed)
    # Set environment options
    env.set_freq(freq)
    env.set_obs_steps(obs_steps)

    keys = ['btcusd', 'ltcusd', 'xrpusd', 'ethusd', 'etcusd', 'xmrusd', 'zecusd', 'iotusd', 'bchusd', 'dshusd', 'stcusd']

    # Add backtest data
    for i in range(n_assets):
        env.add_df(df=dfs[i], symbol=keys[i])
        env.add_symbol(keys[i])
        env.set_init_crypto(init_crypto, keys[i])
        env.set_tax(tax, keys[i])
    del dfs

    env.set_init_fiat(init_fiat)

    # Clean pools
    env._reset_status()
    env.clear_dfs()

    if test:
        env.set_training_stage(False)
    else:
        env.set_training_stage(True)
    env.set_observation_space()
    env.set_action_space()
    env.reset(reset_funds=True, reset_results=True)

    return env


class TrainingEnvironment(Env):
    '''
    The end and the beginning, the revelation of a new life
    '''
    def __init__(self, db=None, name=None, seed=42):

        try:
            assert isinstance(name, str)
            self.name = name
        except AssertionError:
            print("Must enter environment name")
            raise ValueError

        self._seed(seed)

        if not os.path.exists('./logs'):
            os.makedirs('./logs')
        self.logger = Logger(self.name, './logs/')

        self.crypto = {}
        self.fiat = None
        self.prev_posit = {}
        self.posit = {}
        self.prev_val = np.nan
        self.init_crypto = {}
        self.init_fiat = None
        self.tax = {}
        self.symbols = ['fiat']

        self.status = None
        self.np_random = None
        self._is_training = False
        self.epsilon = 1e-8
        self.step_idx = None
        self.global_step = 0
        self.obs_steps = 0
        self.offset = 0
        self.last_reward = 0.0
        self.last_action = np.zeros(0)

        # Data input
        # Gets data table from database server
        self.db = db
        self.tables = {}
        self.dfs = {}
        self.df = None

        self.logger.info("Training Environment initialization",
                         "Training Environment Initialized!")

    # Setters
    # def add_table(self, number=None, name=None):
    #     try:
    #         try:
    #             assert isinstance(self.db, pm.database.Database)
    #         except AssertionError:
    #             print("db must be a pymongo database instance")
    #             raise ValueError
    #
    #         col_names = self.db.collection_names()
    #         col_names_str = ""
    #         for i, n in enumerate(col_names):
    #             col_names_str += "Table %d: %s, Count: %d\n" % (i, n, self.db[n].count())
    #
    #         welcome_str = "Welcome to the Apocalipse trading environment!\n"+\
    #                       "Select from db a table to trade on:\n" + col_names_str
    #
    #         if isinstance(number, int):
    #             table_n = number
    #         elif isinstance(name, str):
    #             table_n = name
    #         else:
    #             print(welcome_str)
    #
    #             table_n = input("Enter a table number or name:")
    #         if isinstance(table_n, int):
    #             table = self.db[col_names[int(table_n)]]
    #             self.logger.info(TrainingEnvironment.add_table,
    #                              "Initializing Apocalipse instance with table %s" % (col_names[int(table_n)]))
    #
    #         if isinstance(table_n, str):
    #             table = self.db[table_n]
    #             self.logger.info(TrainingEnvironment.add_table,
    #                              "Initializing Apocalipse instance with table %s" % (table_n))
    #
    #         if table_n == '':
    #             print("You must enter a table number or name!")
    #             raise ValueError
    #
    #         assert isinstance(table, pm.collection.Collection)
    #         assert table.find_one()['columns'] == ['trade_px', 'trade_volume', 'trades_date_time']
    #
    #         symbol = self._get_table_symbol(table)
    #         self.tables[symbol] = table
    #
    #     except AssertionError:
    #         self.logger.error(TrainingEnvironment.add_table, "Table error. Please, enter a valid table number.")
    #         raise ValueError
    #     except Exception as e:
    #         if debug:
    #             self.logger.error(TrainingEnvironment.add_table, self.parse_error(e))
    #         else:
    #             self.logger.error(TrainingEnvironment.add_table, "Wrong table.")
    #             raise ValueError

    def clear_dfs(self):
        if hasattr(self, 'dfs'):
            del self.dfs
        self.dfs = {}

    def add_df(self, df=None, symbol=None, steps=5):
        try:
            assert isinstance(self.freq, int) and self.freq >=1

            assert isinstance(steps, int) and steps >= 3
            if isinstance(df, pd.core.frame.DataFrame):
                for col in df.columns:
                    assert col in ['open', 'high', 'low', 'close', 'volume', 'prev_position', 'position', 'amount'], \
                    'wrong dataframe formatation'
                self.dfs[symbol] = df.ffill().fillna(1e-8).applymap(convert_to.decimal)
            else:
                assert symbol in [s for s in self.tables.keys()]
                assert isinstance(self.tables[symbol], pm.collection.Collection)

                try:
                    assert steps >= 3
                except AssertionError:
                    print("Observation steps must be greater than 3")
                    return False
                self.dfs[symbol] = self._get_obs(symbol=symbol, steps=steps, freq=self.freq)

            assert isinstance(self.dfs[symbol], pd.core.frame.DataFrame)

            self.dfs[symbol]['prev_position'] = np.nan
            self.dfs[symbol]['position'] = np.nan
            self.dfs[symbol]['amount'] = np.nan

            self.dfs[symbol] = self.dfs[symbol].ffill().fillna(1e-8).applymap(convert_to.decimal)

            assert self.dfs[symbol].columns.all() in ['open', 'high', 'low', 'close', 'volume', 'prev_position',
                                                      'position', 'amount']

        except Exception as e:
            self.logger.error(TrainingEnvironment.add_df, self.parse_error(e))

    def add_symbol(self, symbol):
        assert isinstance(symbol, str)
        self.symbols.append(symbol)
        if symbol not in [k for k in self.dfs.keys()]:
            self.add_df(symbol=symbol, steps=int(self.obs_steps * 5))
        self.make_df()
        self._set_posit(convert_to.decimal('0.0'), symbol, self.df.index[0])

    def make_df(self):
        self.df = pd.concat(self.dfs, axis=1)
        assert isinstance(self.df, pd.core.frame.DataFrame)
        self.df['fiat', 'prev_position'] = convert_to.decimal(np.nan)
        self.df['fiat', 'position'] = convert_to.decimal(np.nan)
        self.df['fiat', 'amount'] = convert_to.decimal(np.nan)
        # self.clear_dfs()
        self.set_observation_space()
        try:
            assert self.df.shape[0] >= self.obs_steps
        except AssertionError:
            self.logger.error(TrainingEnvironment.make_df, "Trying to make dataframe with less observations than obs_steps.")

    def set_init_fiat(self, fiat):
        assert fiat >= 0.0
        self.init_fiat = convert_to.decimal(fiat)

    def set_init_crypto(self, amount, symbol):
        assert amount >= 0.0
        assert symbol in self._get_df_symbols()
        self.init_crypto[symbol] = convert_to.decimal(amount)

    def _set_fiat(self, fiat, timestamp):
        try:
            # assert isinstance(timestamp, pd.Timestamp) # taken out for speed
            assert isinstance(fiat, Decimal), 'fiat is not decimal'

            if fiat < convert_to.decimal('0E-8'):
                self.status['ValueError'] += 1
                self.logger.error(TrainingEnvironment._set_fiat, "Fiat value error: Negative value")
            self.fiat = fiat
            self.df.loc[timestamp, ('fiat', 'amount')] = fiat
        except Exception as e:
            self.logger.error(TrainingEnvironment._set_fiat, self.parse_error(e))

    def _set_crypto(self, amount, symbol, timestamp):
        try:
            # assert symbol in self._get_df_symbols(no_fiat=True) # taken out for speed
            # assert isinstance(timestamp, pd.Timestamp) # taken out for speed
            assert isinstance(amount, Decimal)
            if amount < 0.0:
                self.status['ValueError'] += 1
                self.logger.error(TrainingEnvironment._set_crypto, "Crypto value error: Negative value")

            self.crypto[symbol] = amount
            self.df.loc[timestamp, (symbol, 'amount')] = amount

        except Exception as e:
            self.logger.error(TrainingEnvironment._set_crypto, self.parse_error(e))

    def _set_posit(self, posit, symbol, timestamp):
        # TODO: VALIDATE
        try:
            try:
                assert isinstance(posit, Decimal)
            except AssertionError:
                if isinstance(posit, float):
                    posit = convert_to.decimal(posit)
                else:
                    raise AssertionError
            # assert isinstance(timestamp, pd.Timestamp) # taken out for speed
            assert  convert_to.decimal('0E-8') <= posit <= convert_to.decimal('1.0'), posit
            self.posit[symbol] = posit
            self.df.loc[timestamp, (symbol, 'position')] = posit

        except AssertionError:
            self.status['ValueError'] += 1
            self.logger.error(TrainingEnvironment._set_posit, "Invalid previous position value.")

        except Exception as e:
            self.logger.error(TrainingEnvironment._set_posit, self.parse_error(e))

    def _set_prev_posit(self, posit, symbol, timestamp):
        try:
            try:
                assert isinstance(posit, Decimal)
            except AssertionError:
                if isinstance(posit, float):
                    posit = convert_to.decimal(posit)
                else:
                    raise AssertionError
            # assert isinstance(timestamp, pd.Timestamp) # taken out for speed
            # try:
            #     assert 0.0 <= posit <= 1.0, posit
            # except AssertionError:
            #     posit = np.clip(np.array([posit]), a_min=0.0, a_max=1.0)[0]
            #     self.status['ValueError'] += 1
            #     self.logger.error(Apocalipse._set_prev_posit, "Value error: Position out of range")
            assert convert_to.decimal('0E-8') <= posit <= convert_to.decimal('1.0'), posit
            self.prev_posit[symbol] = posit
            self.df.loc[timestamp, (symbol, 'prev_position')] = posit

        except AssertionError:
            self.status['ValueError'] += 1
            self.logger.error(TrainingEnvironment._set_prev_posit, "Invalid previous position value.")

        except Exception as e:
            self.logger.error(TrainingEnvironment._set_prev_posit, self.parse_error(e))

    def _save_prev_portval(self):
        try:
            portval = self._calc_step_total_portval()
            assert portval >= 0.0
            self.prev_val = portval
        except Exception as e:
            self.logger.error(TrainingEnvironment._save_prev_portval, self.parse_error(e))

    def _save_prev_portifolio_posit(self, timestamp):
        for symbol in self._get_df_symbols():
            self._set_prev_posit(self._get_posit(symbol), symbol, timestamp)

    def set_action_space(self):
        # Action space
        self.action_space = Box(0., 1., len(self._get_df_symbols()))
        self.logger.info(TrainingEnvironment.set_action_space, "Setting environment with %d symbols." % (len(self._get_df_symbols())))

    def set_observation_space(self):
        # Observation space:
        obs_space = []
        # OPEN, HIGH, LOW, CLOSE
        for _ in range(4):
            obs_space.append(Box(0.0, 1e9, 1))
        # VOLUME
        obs_space.append(Box(0.0, 1e12, 1))
        # POSITION
        obs_space.append(Box(0.0, 1.0, 1))

        self.observation_space = Tuple(obs_space)

    def set_obs_steps(self, steps):
        assert isinstance(steps, int) and steps >= 3
        self.obs_steps = steps
        self.step_idx = steps
        self.offset = steps

    def set_freq(self, freq):
        assert isinstance(freq, int) and freq >= 1, "frequency must be a integer >= 1"
        self.freq = freq

    def set_tax(self, tax, symbol):
        assert 0.0 <= tax <= 1.0
        assert symbol in self._get_df_symbols()
        self.tax[symbol] = convert_to.decimal(tax)

    def set_training_stage(self, train):
        assert isinstance(train, bool)
        self._is_training = train

    def _save_df_to_db(self, name):
        self.db[name].insert_one(self.df.applymap(convert_to.decimal128).to_dict(orient='split'))

    # Getters
    def _get_df_symbols(self, no_fiat=False):
        if no_fiat:
            return [s for s in self.df.columns.levels[0] if s is not 'fiat']
        else:
            return [s for s in self.df.columns.levels[0]]

    @staticmethod
    def _get_table_symbol(table):
        return table.full_name.split('.')[1].split('_')[2]

    def get_step_obs_all(self):
        obs_list = []
        keys = []
        for symbol in self._get_df_symbols():
            keys.append(symbol)
            obs_list.append(self.get_step_obs(symbol))

        return pd.concat(obs_list, keys=keys, axis=1)

    def get_step_obs(self, symbol):
        return self._get_step_obs(symbol=symbol, steps=self.obs_steps, float=True)

    def _get_step_obs(self, symbol, steps=1, step_price=False, float=False):
        """
        Observe the environment. Is usually used after the step is taken
        """
        # observation as per observation space
        try:
            # assert symbol in self._get_df_symbols() #taken out for speed
            # assert steps >= 1 and isinstance(last_price, bool) #taken out for speed

            if step_price:
                return self.df.get_value(self.df.index[self.step_idx], (symbol, 'close'))

            else:
                if symbol in self._get_df_symbols(no_fiat=True):
                    columns = ['open', 'high', 'low', 'close', 'volume', 'position']

                else:
                    columns = ['position']

                # obs = self.df.loc[self.df.index[self.step_idx - steps + 1:self.step_idx + 1],
                #                   (symbol, columns)]
                # Better performance methods
                # idx = self.df[symbol].columns.get_indexer(columns)
                # obs = self.df[symbol].iloc[self.step_idx - steps + 1:self.step_idx + 1].take(idx, axis=1)

                obs = self.df[symbol].iloc[self.step_idx - steps + 1:self.step_idx + 1].filter(columns)

                # ffill last position with previous one
                obs.iat[-1,-1] = obs.iat[-2,-1]

                assert obs.shape[0] == self.obs_steps
                if float:
                    return obs.astype(np.float64)
                else:
                    return obs

        except IndexError as e:
            self.logger.error(TrainingEnvironment._get_step_obs, self.parse_error(e))
            return False
        except Exception as e:
            self.logger.error(TrainingEnvironment._get_step_obs, self.parse_error(e))
            return False

    def _get_fiat(self):
        assert self.fiat >= convert_to.decimal('0E-12'), self.fiat
        return self.fiat

    def _get_crypto(self, symbol):
        assert self.crypto[symbol] >= Decimal('0E-12'), self.crypto[symbol]
        # assert symbol in self._get_df_symbols(no_fiat=True) # Took out for sp
        return self.crypto[symbol]

    def _get_init_fiat(self):
        assert convert_to.decimal('0E-8') <= self.init_fiat
        return self.init_fiat

    def _get_init_crypto(self, symbol):
        assert symbol in [s for s in self.init_crypto.keys()], (symbol, self.init_crypto.keys())
        assert 0.0 <= self.init_crypto[symbol], self.init_crypto[symbol]
        return self.init_crypto[symbol]

    def _get_posit(self, symbol):
        assert self.posit[symbol] <= 1.0
        return self.posit[symbol]

    def _get_portifolio_posit(self):
        portifolio = []
        for symbol in self._get_df_symbols():
            portifolio.append(self._get_posit(symbol))
        return np.array(portifolio)

    def _get_prev_posit(self, symbol):
        assert self.prev_posit[symbol] <= 1.0
        return self.prev_posit[symbol]

    def _get_prev_portifolio_posit(self, no_fiat=False):
        portifolio = []
        for symbol in self._get_df_symbols():
            portifolio.append(self._get_prev_posit(symbol))
        if no_fiat:
            return np.array(portifolio)[:-1]
        else:
            return np.array(portifolio)

    def _calc_step_posit(self, symbol):
        if symbol is not 'fiat':
            return self._get_crypto(symbol) * self._get_step_obs(symbol, step_price=True) /\
                   self._calc_step_total_portval()
        else:
            return self._get_fiat() / self._calc_step_total_portval()

    def _calc_step_portifolio_posit(self):
        portifolio = []
        for symbol in self._get_df_symbols():
            portifolio.append(self._calc_step_posit(symbol))
        return np.array(portifolio)

    def _calc_step_portval(self, symbol):
        return self._get_crypto(symbol) * self._get_step_obs(symbol, step_price=True)

    def _calc_step_total_portval(self):
        portval = convert_to.decimal('0.0')

        for symbol in self._get_df_symbols(no_fiat=True):
            portval += self._get_crypto(symbol) * self._get_step_obs(symbol, step_price=True)
        portval += self._get_fiat()

        return portval

    def _get_prev_portval(self):
        assert self.prev_val >= 0.0
        return self.prev_val

    def _get_tax(self, symbol):
        # assert symbol in [s for s in self.tax.keys()]
        return self.tax[symbol]

    def _get_historical_data(self, symbol=None, start=None, end=None, freq=1, file=None):
        """
        Gets obs from data server
        :args:
        :steps: int: Number of bar to retrieve
        :freq: int: Sampling frequency in minutes
        """
        try:
            assert freq >= 1

            start = pd.to_datetime(start)
            end = pd.to_datetime(end)

            if file:
                assert isinstance(file, str) or isinstance(file, pd.core.frame.DataFrame)
                obs = get_historical(file, freq, start, end)
            else:
                assert symbol in [s for s in self.tables.keys()]
                columns = self.tables[symbol].find().limit(-1).sort('index', pm.DESCENDING).next()['columns']

                cursor = self.tables[symbol].find({'index': {'$lt': end,
                                                    '$gte': start}}).sort('index', pm.DESCENDING)

                data = [[item['data'][0][i] for i in range(len(item) - 2)] + [item['data'][0][2]] for item in cursor]

                obs = self.sample_trades(pd.DataFrame(data=data, columns=columns).set_index('trades_date_time',
                                                                                        drop=True),
                                                                                        "%dmin" % (freq))

            obs.ffill(inplace=True)
            obs.bfill(inplace=True)

            try:
                assert obs.shape[0] > 0
            except AssertionError:
                self.logger.info(TrainingEnvironment._get_historical_data,
                                 "There is no data for this time period within the data server.")
                obs = False

            return obs

        except Exception as e:
            self.logger.error(TrainingEnvironment._get_historical_data, self.parse_error(e))
            return False

    def _get_timestamp_now(self):
        return pd.to_datetime(datetime.now() + timedelta(hours=3))  # timezone

    ## Environment methos
    def _get_reward(self, type='absolute return'):
        if type == 'absolute return':
            return self._calc_step_total_portval() - self._get_prev_portval()

        elif type == 'percent change':
            prev_portval = self._get_prev_portval()
            if prev_portval > convert_to.decimal('0.0'):
                return (self._calc_step_total_portval() - self._get_prev_portval()) / prev_portval
            else:
                return (self._calc_step_total_portval() - self._get_prev_portval())

        elif type == 'sharpe ratio':
            # TODO: IMPLEMENT SHARPE REWARD
            raise NotImplementedError
            return self._calc_step_total_portval() - self._get_prev_portval()

        else:
            raise NotImplementedError

    def _assert_action(self, action):
        try:
            for posit in action:
                if not isinstance(posit, Decimal):
                    action = convert_to.decimal(np.float64(action))

            try:
                assert self.action_space.contains(action)

            except AssertionError:
                # normalize
                if action.sum() != convert_to.decimal('1.0'):
                    action /= action.sum()
                    try:
                        assert action.sum() == convert_to.decimal('1.0')
                    except AssertionError:
                        action[-1] += convert_to.decimal('1.0') - action.sum()
                        action /= action.sum()
                        assert action.sum() == convert_to.decimal('1.0')

                # if debug:
                #     self.logger.error(Apocalipse._assert_action, "Action does not belong to action space")

            assert action.sum() - convert_to.decimal('1.0') < convert_to.decimal('1e-6')

        except AssertionError:
            if debug:
                self.status['ActionError'] += 1
                # self.logger.error(Apocalipse._assert_action, "Action out of range")

            action /= action.sum()
            try:
                assert action.sum() == convert_to.decimal('1.0')
            except AssertionError:
                action[-1] += convert_to.decimal('1.0') - action.sum()
                try:
                    assert action.sum() == convert_to.decimal('1.0')
                except AssertionError:
                    action /= action.sum()

        return action

    def _reset_funds(self):
        for symbol in self._get_df_symbols(no_fiat=True):
            self.df[(symbol, 'amount')] = np.nan
            self._set_crypto(self._get_init_crypto(symbol), symbol, self.df.index[self.step_idx])

        self.df['fiat'] = np.nan
        self._set_fiat(self._get_init_fiat(), self.df.index[self.step_idx])

    def _reset_status(self):
        self.status = {'OOD': False, 'Error': False, 'ValueError': False, 'ActionError': False}

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _simulate_trade(self, action, timestamp):

        # Assert inputs
        action = self._assert_action(action)
        # for symbol in self._get_df_symbols(no_fiat=True): TODO FIX THIS
        #     self.observation_space.contains(observation[symbol])
        # assert isinstance(timestamp, pd.Timestamp)

        # Calculate position change given action
        posit_change = (convert_to.decimal(action) - self._calc_step_portifolio_posit())[:-1]

        # Get fiat_pool
        fiat_pool = self._get_fiat()
        portval = self._calc_step_total_portval()

        # Sell assets first
        for i, change in enumerate(posit_change):
            if change < convert_to.decimal('0E-8'):

                symbol = self._get_df_symbols()[i]

                crypto_pool = portval * action[i] / self._get_step_obs(symbol, step_price=True)

                with localcontext() as ctx:
                    ctx.rounding = ROUND_UP

                    fee = portval * change.copy_abs() * self._get_tax(symbol)

                fiat_pool += portval.fma(change.copy_abs(), -fee)

                self._set_crypto(crypto_pool, symbol, timestamp)

        self._set_fiat(fiat_pool, timestamp)

        # Uodate prev portval with deduced taxes
        portval = self._calc_step_total_portval()

        # Then buy some goods
        for i, change in enumerate(posit_change):
            if change > convert_to.decimal('0E-8'):

                symbol = self._get_df_symbols()[i]

                fiat_pool -= portval * change.copy_abs()

                # if fiat_pool is negative, deduce it from portval and clip
                try:
                    assert fiat_pool >= convert_to.decimal('0E-8')
                except AssertionError:
                    portval += fiat_pool
                    fiat_pool = convert_to.decimal('0E-8')
                    # if debug:
                    #     self.status['ValueError'] += 1
                    #     self.logger.error(Apocalipse._simulate_trade,
                    #                       "Negative value for fiat pool at trade end." + str(fiat_pool))

                with localcontext() as ctx:
                    ctx.rounding = ROUND_UP

                    fee = self._get_tax(symbol) * portval * change

                crypto_pool = portval.fma(action[i], -fee) / self._get_step_obs(symbol, step_price=True)

                self._set_crypto(crypto_pool, symbol, timestamp)

        # And don't forget to take your change!
        self._set_fiat(fiat_pool, timestamp)

        # If nothing more interests you, just save and leave
        for i, change in enumerate(posit_change):
            if change == convert_to.decimal('0.0'):
                # No order to execute, just save the variables and exit

                symbol = self._get_df_symbols()[i]

                self._set_crypto(self._get_crypto(symbol), symbol, timestamp)

        ## Update your position on the exit
        for i, symbol in enumerate(self._get_df_symbols(no_fiat=True)):
            self._set_posit(self._get_step_obs(symbol, step_price=True) * self._get_crypto(symbol) /
                            self._calc_step_total_portval(), symbol, timestamp)
        self._set_posit(self._get_fiat() / self._calc_step_total_portval(), 'fiat', timestamp)

    def reset(self, reset_funds=True, reset_results=False, reset_global_step=False):
        """
        Resets environment and returns a initial observation
        :param reset_funds: Reset funds to initial value
        :param reset_results: Reset results from instance dataframe
        :param reset_global_step: Reset global step counter
        :return: observation
        """

        if reset_global_step:
            self.global_step = 0

        if self._is_training:
            self.step_idx = self.offset + np.random.randint(high=self.df.shape[0] - self.offset - 1, low=0)
            timestamp = self.df.index[self.step_idx]
        else:
            self.step_idx = self.offset
            timestamp = self.df.index[self.step_idx]

        for symbol in self._get_df_symbols():
            self._reset(symbol, timestamp, reset_funds=reset_funds, reset_results=reset_results)

        self.step_idx -= self.obs_steps

        for _ in range(self.obs_steps):
            timestamp = self.df.index[self.step_idx]

            for symbol in self._get_df_symbols():
                # # Set crypto and fiat amounts
                if symbol is not 'fiat':
                    self._set_crypto(self._get_crypto(symbol), symbol, timestamp)
                else:
                    self._set_fiat(self._get_fiat(), timestamp)

                # portval = self._calc_step_total_portval()

                # Calculate positions
                # if portval > convert_to.decimal('0.0'):
                posit = self._calc_step_posit(symbol)
                self._set_prev_posit(posit, symbol, timestamp)
                self._set_posit(posit, symbol, timestamp)
                # else:
                #     self._set_prev_posit(convert_to.decimal('0.0'), symbol, timestamp)
                #     self._set_posit(convert_to.decimal('0.0'), symbol, timestamp)

            self.step_idx += 1

        assert (self.crypto and self.fiat and self.obs_steps and self.offset and self.tax) is not None

        return self.get_step_obs_all()

    def _reset(self, symbol, timestamp, reset_results=False, reset_funds=False):
        # TODO: VALIDATE
        # Assert conditions
        assert symbol in self._get_df_symbols()
        assert (self._is_training, self.obs_steps, self.tax) is not None
        assert isinstance(self.obs_steps, int) and self.obs_steps > 0

        try:
            assert self.df[symbol].shape[0] > self.obs_steps
        except AssertionError:
            self.logger.error(TrainingEnvironment._reset, "Calling reset with steps <= obs_steps")
            raise ValueError

        # set offset
        self.offset = self.obs_steps

        # Reset results columns
        if reset_results:
            self.df[symbol, 'amount'] = convert_to.decimal(np.nan)
            self.df[symbol, 'position'] = convert_to.decimal(np.nan)
            self.df[symbol, 'prev_position'] = convert_to.decimal(np.nan)

        if symbol is not 'fiat':
            if len(self.crypto.keys()) <= len(self._get_df_symbols()) - 1 or reset_funds or self._is_training:
                self._set_crypto(self._get_init_crypto(symbol), symbol, timestamp)
            else:
                self._set_crypto(self._get_crypto(symbol), symbol, timestamp)
        else:
            if self.fiat is None or reset_funds or self._is_training:
                self._set_fiat(self._get_init_fiat(), timestamp)
            else:
                self._set_fiat(self._get_fiat(), timestamp)

        # self.logger.info(Apocalipse._reset, "Symbol %s reset done." % (symbol))

        # return self._get_step_obs(symbol, self.obs_steps)

    def step(self, action):
        return self._step(self.get_step_obs_all(), action)

    def _step(self, observation, action, reward_type='percent change', timeout=10):
        # TODO: VALIDATE
        try:
            # Assert observation is valid
            # assert (isinstance(observation, pd.core.frame.DataFrame) or isinstance(observation, np.ndarray)) and \
            #        observation.shape[0] >= 3

            # for symbol in self._get_df_symbols(no_fiat=True):
            #     assert self.observation_space.contains(observation[symbol])

            # Assert action is valid
            action = self._assert_action(action)
            self.last_action = action

            # Get step timestamp
            timestamp = self.df.index[self.step_idx]

            # Save previous val and position
            self._save_prev_portval()
            self._save_prev_portifolio_posit(timestamp)

            self._simulate_trade(action, timestamp)

            # Update step counter and environment observation
            if self.step_idx < self.df.shape[0] - 1:
                self.step_idx += 1
                done = False
            else:
                self.status['OOD'] = True
                done = True

            new_obs = self.get_step_obs_all()

            # Calculate step reward
            self.last_reward = self._get_reward(reward_type)

            if isinstance(new_obs, pd.core.frame.DataFrame):
                assert new_obs.shape[0] == observation.shape[0], "wrong observation size"
                assert pd.infer_freq(new_obs.index) == pd.infer_freq(observation.index), "wrong observation frequency"
                # assert len(new_obs.index) == len(observation.index), "wrong observation size"

            self.global_step += 1

            return new_obs, np.float32(self.last_reward), done, self.status

        except Exception as e:
            self.logger.error(TrainingEnvironment._step, self.parse_error(e))
            self._send_email(self.name + "step error:", self.parse_error(e))

            return False, False, True, self.status

    # Helper methods
    def parse_error(self, e):
        error_msg = '\n' + self.name + ' error -> ' + type(e).__name__ + ' in line ' + str(
            e.__traceback__.tb_lineno) + ': ' + str(e)
        return error_msg

    def set_email(self, email, psw):
        """
        Set Gmail address and password for log keeping
        :param email: str: Gmail address
        :param psw: str: account password
        :return:
        """
        try:
            assert isinstance(email, str) and isinstance(psw, str)
            self.email = email
            self.psw = psw
            self.logger.info(TrainingEnvironment.set_email, "Email report address set to: %s" % (self.email))
        except Exception as e:
            self.logger.error(TrainingEnvironment.set_email, self.parse_error(e))

    def _send_email(self, subject, body):
        try:
            assert isinstance(self.email, str) and isinstance(self.psw, str) and \
                   isinstance(subject, str) and isinstance(body, str)
            gmail_user = self.email
            gmail_pwd = self.psw
            FROM = self.email
            TO = self.email if type(self.email) is list else [self.email]
            SUBJECT = subject
            TEXT = body

            # Prepare actual message
            message = """From: %s\nTo: %s\nSubject: %s\n\n%s
                    """ % (FROM, ", ".join(TO), SUBJECT, TEXT)

            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.ehlo()
            server.starttls()
            server.login(gmail_user, gmail_pwd)
            server.sendmail(FROM, TO, message)
            server.close()

        except Exception as e:
            self.logger.error(TrainingEnvironment._send_email, self.parse_error(e))
        finally:
            pass

    def plot_results(self, df):
        def config_fig(fig):
            fig.background_fill_color = "black"
            fig.background_fill_alpha = 0.5
            fig.border_fill_color = "#232323"
            fig.outline_line_color = "#232323"
            fig.title.text_color = "whitesmoke"
            fig.xaxis.axis_label_text_color = "whitesmoke"
            fig.yaxis.axis_label_text_color = "whitesmoke"
            fig.yaxis.major_label_text_color = "whitesmoke"
            fig.xaxis.major_label_orientation = np.pi / 4
            fig.grid.grid_line_alpha = 0.3

        df = df.astype(np.float64)
        # Results figures
        results = {}

        # Position
        p_pos = figure(title="Position over time",
                       x_axis_type="datetime",
                       x_axis_label='timestep',
                       y_axis_label='position',
                       plot_width=800, plot_height=400,
                       tools='crosshair,hover,reset,xwheel_zoom,pan,box_zoom',
                       toolbar_location="above"
                       )
        config_fig(p_pos)

        palettes = inferno(len(self._get_df_symbols()))

        for i, symbol in enumerate(self._get_df_symbols()):
            results[symbol + '_posit'] = p_pos.line(df.index, df[symbol, 'position'], color=palettes[i], legend=symbol)

        # Portifolio and benchmark values
        p_val = figure(title="Portifolio / Benchmark Value",
                       x_axis_type="datetime",
                       x_axis_label='timestep',
                       y_axis_label='position',
                       plot_width=800, plot_height=400,
                       tools='crosshair,hover,reset,xwheel_zoom,pan,box_zoom',
                       toolbar_location="above"
                       )
        config_fig(p_val)

        results['portval'] = p_val.line(df.index, df.portval, color='green')
        results['benchmark'] = p_val.line(df.index, df.benchmark, color='red')

        # Portifolio and benchmark returns
        p_ret = figure(title="Portifolio / Benchmark Returns",
                       x_axis_type="datetime",
                       x_axis_label='timestep',
                       y_axis_label='Returns',
                       plot_width=800, plot_height=200,
                       tools='crosshair,hover,reset,xwheel_zoom,pan,box_zoom',
                       toolbar_location="above"
                       )
        config_fig(p_ret)

        results['bench_ret'] = p_ret.line(df.index, df.benchmark_returns, color='red')
        results['port_ret'] = p_ret.line(df.index, df.returns, color='green')
        
        p_hist = figure(title="Portifolio Value Pct Change Distribution",
                        x_axis_label='Pct Change',
                        y_axis_label='frequency',
                        plot_width=800, plot_height=300,
                        tools='crosshair,hover,reset,xwheel_zoom,pan,box_zoom',
                        toolbar_location="above"
                        )
        config_fig(p_hist)

        hist, edges = np.histogram(df.returns, density=True, bins=100)

        p_hist.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
                    fill_color="#036564", line_color="#033649")

        # Portifolio rolling alpha
        p_alpha = figure(title="Portifolio rolling alpha",
                         x_axis_type="datetime",
                         x_axis_label='timestep',
                         y_axis_label='alpha',
                         plot_width=800, plot_height=200,
                         tools='crosshair,hover,reset,xwheel_zoom,pan,box_zoom',
                         toolbar_location="above"
                         )
        config_fig(p_alpha)

        results['alpha'] = p_alpha.line(df.index, df.alpha, color='yellow')

        # Portifolio rolling beta
        p_beta = figure(title="Portifolio rolling beta",
                        x_axis_type="datetime",
                        x_axis_label='timestep',
                        y_axis_label='beta',
                        plot_width=800, plot_height=200,
                        tools='crosshair,hover,reset,xwheel_zoom,pan,box_zoom',
                        toolbar_location="above"
                        )
        config_fig(p_beta)

        results['beta'] = p_beta.line(df.index, df.beta, color='yellow')

        # Rolling Drawdown
        p_dd = figure(title="Portifolio rolling drawdown",
                      x_axis_type="datetime",
                      x_axis_label='timestep',
                      y_axis_label='drawdown',
                      plot_width=800, plot_height=200,
                      tools='crosshair,hover,reset,xwheel_zoom,pan,box_zoom',
                      toolbar_location="above"
                      )
        config_fig(p_dd)

        results['drawdown'] = p_dd.line(df.index, df.drawdown, color='red')

        # Portifolio Sharpe ratio
        p_sharpe = figure(title="Portifolio rolling Sharpe ratio",
                          x_axis_type="datetime",
                          x_axis_label='timestep',
                          y_axis_label='Sharpe ratio',
                          plot_width=800, plot_height=200,
                          tools='crosshair,hover,reset,xwheel_zoom,pan,box_zoom',
                          toolbar_location="above"
                          )
        config_fig(p_sharpe)

        results['sharpe'] = p_sharpe.line(df.index, df.sharpe, color='yellow')

        print("################### > Portifolio Performance Analysis < ###################\n")
        print("Portifolio excess Sharpe:                 %f" % ec.excess_sharpe(df.returns, df.benchmark_returns))
        print("Portifolio / Benchmark Sharpe ratio:      %f / %f" % (ec.sharpe_ratio(df.returns),
                                                                     ec.sharpe_ratio(df.benchmark_returns)))
        print("Portifolio / Benchmark Omega ratio:       %f / %f" % (ec.omega_ratio(df.returns),
                                                                     ec.omega_ratio(df.benchmark_returns)))
        print("Portifolio / Benchmark max drawdown:      %f / %f" % (ec.max_drawdown(df.returns),
                                                                     ec.max_drawdown(df.benchmark_returns)))

        results['handle'] = show(column(p_val, p_pos, p_ret, p_hist, p_sharpe, p_dd, p_alpha, p_beta), notebook_handle=True)

        return results

    def _get_results(self, window=30):
        """
        Calculate arbiter desired actions statistics
        :return:
        """

        self.results = self.df.iloc[self.offset + 1:].copy()

        self.results['portval'] = self.results['fiat', 'amount']
        self.results['benchmark'] = convert_to.decimal('0e-8')
        self.results['returns'] = convert_to.decimal(np.nan)
        self.results['benchmark_returns'] = convert_to.decimal(np.nan)
        self.results['alpha'] = convert_to.decimal(np.nan)
        self.results['beta'] = convert_to.decimal(np.nan)
        self.results['drawdown'] = convert_to.decimal(np.nan)
        self.results['sharpe'] = convert_to.decimal(np.nan)

        for symbol in self._get_df_symbols(no_fiat=True):
            self.results[symbol + '_portval'] = self.results[symbol, 'close'] * self.results[symbol, 'amount']
            self.results['portval'] = self.results['portval'] + self.results[symbol + '_portval']

        # self.results['benchmark'] = self.results['btcusd', 'close'] * self._get_init_fiat() / self.df.at[
        #                             self.results.index[self.offset], ('btcusd', 'close')] - \
        #                             self._get_tax('btcusd') * self._get_init_fiat() / self.results.at[
        #                             self.results.index[self.offset], ('btcusd', 'close')]

        # Calculate benchmark portifolio, just equaly distribute money over all the assets
        for symbol in self._get_df_symbols(no_fiat=True):
            self.results[symbol+'_benchmark'] = (1 - self._get_tax(symbol)) * self.results[symbol, 'close'] * \
                                        self._get_init_fiat() / (self.results.at[self.results.index[0],
                                        (symbol, 'close')] * (self.action_space.low.shape[0] - 1))
            self.results['benchmark'] = self.results['benchmark'] + self.results[symbol + '_benchmark']

        self.results['returns'] = pd.to_numeric(self.results.portval).diff().fillna(1e-8)
        self.results['benchmark_returns'] = pd.to_numeric(self.results.benchmark).diff().fillna(1e-8)
        self.results['alpha'] = ec.utils.roll(self.results.returns,
                                              self.results.benchmark_returns,
                                              function=ec.alpha_aligned,
                                              window=window,
                                              risk_free=0.001
                                              )
        self.results['beta'] = ec.utils.roll(self.results.returns,
                                             self.results.benchmark_returns,
                                             function=ec.beta_aligned,
                                             window=window)
        self.results['drawdown'] = ec.roll_max_drawdown(self.results.returns, window=int(window/10))
        self.results['sharpe'] = ec.roll_sharpe_ratio(self.results.returns, window=int(window/5), risk_free=0.001)

        return self.results

    # TODO: \/ ############### REWORK ################# \/

    def render(self, mode='human', close=False):
        return self._render(close=close)

    def _render(self, mode='human', close=False):
        if close:
            return
        if mode == 'human':
            print("""\n>> Step: {0}, \
                     \n   Timestamp: {1}, \
                     \n   Crypto Price: {2}, \
                     \n   Fiat allocation: {3}, \
                     \n   Previous position: {4}, \
                     \n   New position (Action): {5}, \
                     \n   Portifolio dolar value: {6}""".format(self.step_idx,
                                                                self.df.index[self.step_idx],
                                                                self._get_obs(last_price=True),
                                                                str(self._get_fiat()),
                                                                str(self._get_prev_portifolio_posit()),
                                                                str(self._get_portifolio_posit()),
                                                                str(self._calc_step_total_portval())))
        if mode == 'print':
            pass

        else:
            raise TypeError


class OLDTradingEnvironment(Env):
    """
    Online environment for automated trading strategies
    """

    # TODO: \/ ################ REWORKING NOW ###################### \/

    # def _rebalance_portifolio(self, action, timestamp, timeout):
    #
    #     order = self._parse_order(action, symbol)
    #
    #     if order and isinstance(order, tuple):
    #         # Send order to exchange
    #         response = self._place_order(*order, timeout)
    #
    #         # Update variables with new exchange state
    #         self._set_crypto(self._get_exchange_crypto(symbol), symbol, timestamp)
    #         self._set_fiat(self._get_exchange_fiat(symbol),  timestamp)
    #     else:
    #
    #         response = False
    #         self._set_crypto(self._get_crypto(symbol), symbol, timestamp)
    #         self._set_fiat(self._get_fiat(symbol), timestamp)
    #
    #     # TODO: VALIDATE
    #     # Update online position
    #     self._set_posit(self._get_posit(symbol), symbol, timestamp)
    #
    #     return response

    def __init__(self, db=None, name=None, seed=42):

        try:
            assert isinstance(name, str)
            self.name = name
        except AssertionError:
            print("Must enter environment name")
            raise ValueError

        self._seed(seed)

        self.online = False

        if not os.path.exists('./online_logs'):
            os.makedirs('./online_logs')
        self.logger = Logger(self.name, './online_logs/')

        self.crypto = {}
        self.fiat = None
        self.prev_posit = {}
        self.posit = {}
        self.prev_val = np.nan
        self.tax = {}
        self.symbols = ['fiat']

        self.status = None
        self.epsilon = 1e-8
        self.step_idx = None
        self.obs_steps = 0
        self.session_begin_time = None
        self.last_reward = 0.0
        self.last_action = np.zeros(0)

        # Data input
        # Gets data table from database server
        self.db = db
        self.tables = {}
        self.dfs = {}
        self.df = None

        self.logger.info("Online Trading Environment initialization",
                         "Trading Environment Initialized!\nONLINE MODE: False\n")


    def _order_done(self, response):
        """
        Find out if order got executed
        :param response: dict: exchange response
        :return: bool: True if order get executed, False if not or error
        """
        try:
            done_orders = self.tapi.user_transactions(limit=int(5 * len(self.tables)))

            for order in done_orders:
                if 'order_id' in order.keys():
                    if int(response['id']) == order['order_id']:
                        return True
                    else:
                        return False
        except Exception as e:
            self.logger.error(TradingEnvironment._order_done, self.parse_error(e))
            self._send_email(self.name + "step error:", self.parse_error(e))
            return False

    def _parse_order(self, action, symbol):
        # TODO: VALIDATE
        try:
            assert self.online and symbol in self._get_df_symbols(no_fiat=True)

            base, quote = symbol[:3], symbol[3:]
            assert isinstance((base and quote), str)

            # Get exchange data
            ticker = self.tapi.ticker_hour(base=base, quote=quote)
            balance = self.tapi.account_balance(base=base, quote=quote)

            # Assert adequate conditions
            if isinstance(action, float):
                action = np.array([action])

            assert isinstance(action, np.ndarray) and 0.0 <= action[0] <= 1.0
            assert isinstance(ticker, dict)
            assert isinstance(balance, dict)

            # Calculate actual position

            with localcontext() as ctx:
                ctx.rounding = ROUND_DOWN

                portval = convert_to.decimal(balance[base + '_balance']) * \
                                 convert_to.decimal(ticker['bid']) + convert_to.decimal(balance[quote + '_balance'])

            posit = convert_to.decimal('1.0') - convert_to.decimal(
                balance[quote + '_balance']) / portval

            with localcontext() as ctx:
                ctx.rounding = ROUND_UP

                fee = convert_to.decimal('1.0') - convert_to.decimal(balance['fee']) / convert_to.decimal('100.0')

            # Calculate position change
            posit_change = convert_to.decimal(action[0]) - posit

            assert -1.0 <= posit_change <= 1.0

            # Get price, amount and side for order placing
            if posit_change < 0.0 and abs(posit_change * portval) > 5:
                side = 'sell'
                price = convert_to.decimal(ticker['bid']) - convert_to.decimal('1e-2')
                amount = convert_to.decimal(
                    abs(posit_change * portval / (price * convert_to.decimal(1.005))))

            elif posit_change > 0.0 and abs(posit_change * portval) > 5:
                side = 'buy'
                price = convert_to.decimal(ticker['ask']) + convert_to.decimal('1e-2')
                amount = convert_to.decimal(
                    abs(fee * posit_change * portval / (price * convert_to.decimal(1.005))))

            else:
                return False

            assert price >= 0 and amount >= 0

            order_report = "Order report:\nDesired Position: %f\nPosition Change: %f\n" % (
            action[0], posit_change) + \
                           "Crypto Price: %f\nOrder Volume: %f\nTimestamp: %s" % (
                               price, amount, str(pd.to_datetime(datetime.utcnow())))

            self.logger.info(TradingEnvironment._parse_order, order_report)

            with localcontext() as ctx:
                ctx.rounding = ROUND_DOWN

                return price.quantize(Decimal('1e-2')), amount.quantize(Decimal('1e-6')), side, symbol

        except Exception as e:
            self.logger.error(TradingEnvironment._parse_order, self.parse_error(e))
            self._send_email("PARSE ORDER ERROR", self.parse_error(e))

    def _place_order(self, price, amount, side, symbol, timeout):
        """
        ONLINE PLACE ORDER FUNCTION
        THIS FUNCTION MAKE ONLINE MOVEMENTS WITH REAL MONEY. USE WITH CAUTION!!!

        :param price: float: order price
        :param amount: float: order volume
        :param side: str: order side, whether buy or sell
        :param symbol: str: pair to trade, btc_usd, btc_eur, ltc_usd, ltc_eur, xrp_usd, xrp_eur
        :param timeout: float: order timeout in seconds
        :return: bool: True if executed, False otherwise
        """
        try:
            # TODO: VALIDATE
            # Initialize order status and timer
            executed = False
            order_timeout = False
            order_time = time()

            base, quote = symbol[:3], symbol[3:]

            open_orders = self.tapi.open_orders(base=base, quote=quote)

            assert isinstance(open_orders, list)
            assert 0 <= price <= 1e8
            assert 0.0 <= amount <= 1e8
            assert side in ['buy', 'sell']
            assert symbol in self._get_df_symbols(no_fiat=True)
            assert isinstance(timeout, float) or isinstance(timeout, int)

            self.logger.info(TradingEnvironment._place_order,
                             "Placing order:\nPrice: %f\nVolume: %f\nSide: %s\nSymbol: %s" % (
                                 price, amount, side, symbol
                             ))

            if side == 'buy':
                response = self.tapi.buy_limit_order(amount, price, base=base, quote=quote)
            elif side == 'sell':
                response = self.tapi.sell_limit_order(amount, price, base=base, quote=quote)
            else:
                self.status['OnlineActionError'] += 1
                self.logger.error(TradingEnvironment._place_order, self.name + " order error: Invalid order side.")
                self._send_email(self.name + " ORDER ERROR", "Invalid order side.")
                return False

            if isinstance(response, dict):
                while not (executed or order_timeout):
                    # Get active orders
                    open_orders = self.tapi.open_orders()

                    if isinstance(open_orders, list):
                        if time() - order_time < float(timeout):
                            if len(open_orders) != 0:
                                if response['id'] in [order['id'] for order in open_orders]:
                                    sleep(2)
                                    continue
                                else:
                                    if self._order_done(response):
                                        executed = True
                                        self.logger.info(TradingEnvironment._place_order, self.name + \
                                                         " order executed: %s" % (str(response)))
                                        self._send_email(self.name + " ORDER EXECUTED",
                                                         "Order executed: %s" % (str(response)))
                            else:
                                if self._order_done(response):
                                    executed = True
                                    self.logger.info(TradingEnvironment._place_order,
                                                     self.name + " order executed: %s" % (str(response)))
                                    self._send_email(self.name + " ORDER EXECUTED",
                                                     "Order executed: %s" % (str(response)))
                                else:
                                    # Order is gone, log and exit
                                    self.status['OnlineActionError'] += 1
                                    self.logger.error(TradingEnvironment._place_order,
                                                      self.name + " order error: Order is not open nor executed," + \
                                                      "trying to cancel: %s" % (str(response)))
                                    self._send_email(self.name + " ORDER ERROR",
                                                     "Order is not open nor executed," + \
                                                     "trying to cancel: %s" % (str(response)))

                                    cancel_response = self.tapi.cancel_order(response['id'])

                                    self.logger.error(TradingEnvironment._place_order,
                                                      self.name + " order error: Order cancel response: %s" % (
                                                          str(cancel_response)))
                                    self._send_email(self.name + " ORDER ERROR",
                                                     "Order cancel response: %s" % (str(cancel_response)))
                                    return executed
                        else:
                            # Cancel order
                            self.logger.info(TradingEnvironment._place_order,
                                             self.name + " order timed out: %s" % (str(response)))
                            try:

                                cancel_response = self.tapi.cancel_order(response['id'])
                                if cancel_response:
                                    order_timeout = True
                                    self.logger.info(TradingEnvironment._place_order,
                                                     self.name + " timed out order canceled: %s" % (str(response)))
                                else:
                                    while not cancel_response:
                                        self.logger.info(TradingEnvironment._place_order, self.name + \
                                                         " cancel order not executed, retrying: %s" % (
                                                         str(response)))
                                        cancel_response = self.tapi.cancel_order(response['id'])
                                        sleep(1)
                                    else:
                                        order_timeout = True
                                        self.logger.info(TradingEnvironment._place_order,
                                                         self.name + " timed out order canceled: %s" % (
                                                         str(response)))

                            except error.BitstampError as e:
                                sleep(2)
                                if self._order_done(response):
                                    executed = True
                                    self.logger.info(TradingEnvironment._place_order,
                                                     self.name + " order timed out " + \
                                                     "but got executed before cancel signal." + \
                                                     " Exchange response: %s" % (str(response)))
                                    self._send_email(self.name + " ORDER EXECUTED",
                                                     "Order timed out but got " + \
                                                     "executed before cancel signal: %s" % (str(response)))
                                else:
                                    self.logger.error(TradingEnvironment._place_order,
                                                      self.name + " order error: Order cancel response: %s" % (
                                                          self.parse_error(e)))
                                    self._send_email(self.name + " ORDER ERROR",
                                                     "Order cancel response: %s" % (self.parse_error(e)))
                                return executed

                    else:
                        self.status['OnlineActionError'] += 1
                        self.logger.error(TradingEnvironment._place_order, self.name + \
                                          " order error: Invalid open orders response: %s" % (str(open_orders)))
                        self._send_email(self.name + " ORDER ERROR",
                                         "Invalid open orders response: %s" % (str(open_orders)))
                        return executed

                else:
                    return executed

        except Exception as e:
            self.logger.error(TradingEnvironment._place_order, self.parse_error(e))
            self._send_email(self.name + " ORDER ERROR", self.parse_error(e))
            return False
        finally:
            pass

            ## Online Methods
            # Setters

    def set_online(self, online, tapi=None, trading_freq=5):
        try:
            assert isinstance(online, bool)
            self.online = online
            self.session_begin_time = None

            # self._get_obs_space()

            if online:
                assert isinstance(tapi, Trading)
                assert trading_freq >= 1

                # self.make_df()
                self.fiat = None
                self.crypto = {}
                self.prev_posit = {}
                self.posit = {}
                self.portifolio_posit = {}

                for symbol in self._get_df_symbols(no_fiat=True):
                    self.crypto[symbol] = Decimal('NaN')
                    self.prev_posit[symbol] = Decimal('NaN')
                    self.posit[symbol] = Decimal('NaN')

                    self.df[symbol, 'prev_position'] = convert_to.decimal(np.nan)
                    self.df[symbol, 'position'] = convert_to.decimal(np.nan)
                    self.df[symbol, 'amount'] = convert_to.decimal(np.nan)

                self.df['fiat', 'fiat'] = convert_to.decimal(np.nan)

                self.tapi = tapi

                timestamp = self.df.index[-1]

                for symbol in self._get_df_symbols(no_fiat=True):
                    amount = self._get_exchange_crypto(symbol)
                    self._set_crypto(amount, symbol, timestamp)
                    self._set_crypto(amount, symbol, timestamp)

                fiat = self._get_exchange_fiat('btcusd')
                self._set_fiat(fiat, timestamp)
                self._set_fiat(fiat, timestamp)

                for symbol in self._get_df_symbols(no_fiat=True):
                    posit = self._get_exchange_posit(symbol)
                    self._set_prev_posit(posit, symbol, timestamp)
                    self._set_posit(posit, symbol, timestamp)
                    self._set_prev_posit(posit, symbol, timestamp)
                    self._set_posit(posit, symbol, timestamp)
                    self.set_tax(convert_to.decimal(self._get_balance(symbol)['fee']) /
                                 convert_to.decimal('100.0'), symbol)

                    assert isinstance(self._get_posit(symbol), Decimal)

                self.session_begin_time = timestamp

                assert isinstance(self._get_ticker(), dict)
                assert isinstance(self._get_portval(), Decimal)

                self.logger.info(TradingEnvironment.set_online, "Trading Environment setup to ONLINE MODE!")

                print(
                    ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ONLINE MODE ON <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                print(
                    "############################################# DISCLAIMER ############################################")
                print(
                    "## In online mode, the steps will be performed on the real exchange environment, with real values! ##")
                print(
                    "## This software comes as is and with no guarantees of any kind. USE AT YOUR OWN RISK!!!!!         ##")
                print(
                    "#####################################################################################################")

        except Exception as e:
            self.logger.error(TradingEnvironment.set_online, self.parse_error(e))

    def set_freq(self, freq):
        assert isinstance(freq, int) and freq >= 1, "frequency must be a integer >= 1"
        self.freq = freq

    def _set_fiat(self, fiat, timestamp):
        """
        Save online fiat volume to Apocalipse instance dataframe
        :param fiat: float: volume to save
        :param timestamp: pandas Timestamp: Timestamp index to save crypto to
        :return:
        """
        # TODO: VALIDATE
        try:
            assert self.online
            try:
                assert isinstance(fiat, Decimal)
            except AssertionError:
                if isinstance(fiat, float):
                    fiat = convert_to.decimal(fiat)
                else:
                    raise AssertionError
            assert isinstance(timestamp, pd.Timestamp)
            assert fiat >= Decimal('0.0')

            self.fiat = fiat
            self.df.at[timestamp, ('fiat', 'fiat')] = fiat
        except Exception as e:
            self.logger.error(TradingEnvironment._set_fiat, self.parse_error(e))

    def _set_crypto(self, amount, symbol, timestamp):
        """
        Save online crypto volume to Apocalipse instance dataframe
        :param crypto: float: volume to save
        :param timestamp: pandas Timestamp: Timestamp index to save crypto to
        :return:
        """
        # TODO: VALIDATE
        try:
            assert self.online
            try:
                assert isinstance(amount, Decimal)
            except AssertionError:
                if isinstance(amount, float):
                    amount = convert_to.decimal(amount)
                else:
                    raise AssertionError
            assert isinstance(timestamp, pd.Timestamp)

            assert amount >= Decimal('0.0')

            self.crypto[symbol] = amount
            self.df.at[timestamp, (symbol, 'amount')] = amount
        except Exception as e:
            self.logger.error(TradingEnvironment._set_crypto, self.parse_error(e))

    def _set_posit(self, posit, symbol, timestamp):
        # TODO: VALIDATE
        try:
            try:
                assert isinstance(posit, Decimal)
            except AssertionError:
                if isinstance(posit, float):
                    posit = convert_to.decimal(posit)
                else:
                    raise AssertionError
            assert isinstance(timestamp, pd.Timestamp)

            assert convert_to.decimal('0.0') <= posit <= convert_to.decimal('1.0')

            self.posit[symbol] = posit
            self.df.at[timestamp, (symbol, 'position')] = posit

        except Exception as e:
            self.logger.error(TradingEnvironment._set_posit, self.parse_error(e))

    def _set_prev_posit(self, posit, symbol, timestamp):
        try:
            try:
                assert isinstance(posit, Decimal)
            except AssertionError:
                if isinstance(posit, float):
                    posit = convert_to.decimal(posit)
                else:
                    raise AssertionError
            assert isinstance(timestamp, pd.Timestamp)

            assert convert_to.decimal('0.0') <= posit <= convert_to.decimal('1.0')

            self.prev_posit[symbol] = posit
            self.df.at[timestamp, (symbol, 'prev_position')] = posit

        except Exception as e:
            self.logger.error(TradingEnvironment._set_prev_posit, self.parse_error(e))

    def _save_prev_portfolio_posit(self, timestamp):
        assert self.online
        for symbol in self._get_df_symbols(no_fiat=True):
            self._set_prev_posit(self._get_posit(symbol), symbol, timestamp)

    def _save_prev_portval(self):
        assert self._get_portval() >= 0.0
        self.prev_val = self._get_portval()

    # TODO VALIDATE _set_obs
    def _set_obs(self, obs):
        """
        Set observation in Apocalipse's instance dataframe
        :param obs: pandas DataFrame: Observation to save
        :return:
        """
        try:
            assert isinstance(obs, pd.DataFrame)

            for symbol in self._get_df_symbols(no_fiat=True):
                if hasattr(obs[symbol], 'position'):
                    del obs[symbol, 'position']
                if hasattr(obs[symbol], 'prev_position'):
                    del obs[symbol, 'prev_position']
            columns = obs.columns.levels[0]
            is_new = False
            # new_df = self.df[symbol].copy()
            for i in obs.index:
                try:
                    self.df = self.df.append(obs.at[i].apply(convert_to.decimal), verify_integrity=True)
                    is_new = True
                except ValueError:
                    for symbol in self._get_df_symbols(no_fiat=True):
                        self.df.at[i, symbol].update(obs.at[i, symbol].apply(convert_to.decimal))

            self.df.asfreq('%dmin' % (self.freq))

            # if is_new:
            #     self.df[symbol] = self.df[symbol].sort_index()

        except Exception as e:
            self.logger.error(TradingEnvironment._set_obs, self.parse_error(e))
        finally:
            pass

    def set_order_type(self, order_type):
        self.order_type = order_type

    # Getters
    def _get_ticker(self, last_price=False):
        """
        Get updated ticker data from exchange
        :param last_price:
        :return:
        """
        try:
            assert self.online

            obs = self.tapi.ticker_hour()

            if last_price:
                return convert_to.decimal(obs['last'])
            else:
                return obs

        except IndexError:
            return False
        except Exception as e:
            self.status['OnlineValueError'] += 1
            self.logger.error(TradingEnvironment._get_ticker, self.parse_error(e))
            return False
        finally:
            pass

    def get_obs_all(self):
        obs_dict = []
        keys = []
        for symbol in self.tables.keys():
            keys = keys.append(symbol)
            obs_dict.append(self.get_obs(symbol))

        return pd.concat(obs_dict, keys=keys, axis=1).ffill()

    def get_obs(self, symbol):
        return self._get_obs(symbol=symbol, steps=self.obs_steps, freq=self.freq)

    def _get_obs(self, symbol=None, steps=1, freq=1, last_price=False, last_obs=False):
        """
        Gets obs from data server
        :args:
        :steps: int: Number of bar to retrieve
        :freq: int: Sampling frequency in minutes
        """
        try:
            assert symbol in self.tables.keys()
            # assert isinstance(self.tables[symbol], pm.collection.Collection) # taken out for speed


            if last_price:
                obs = self.tables[symbol].find().limit(-1).sort('index', pm.DESCENDING).next()['data'][0][0]

            elif last_obs:
                cursor = self.tables[symbol].find().limit(3).sort('index', pm.DESCENDING)
                columns = cursor.next()['columns']

                data = [[item['data'][0][i] for i in range(len(item) - 2)] + [item['data'][0][2]] for item in cursor]

                obs = sample_trades(pd.DataFrame(data=data,
                                            columns=columns).set_index('trades_date_time', drop=True), "%dmin" % (freq))

            else:
                obs = False
                df_steps = steps
                while not isinstance(obs, pd.core.frame.DataFrame):
                    assert isinstance(df_steps, int) and isinstance(freq, int) and isinstance(last_price, bool)
                    assert df_steps >= 1 and freq >= 1

                    offset = int(df_steps * freq)
                    end = datetime.now() + timedelta(hours=3)
                    start = datetime.now() + timedelta(hours=3) - timedelta(minutes=offset, seconds=1)

                    columns = self.tables[symbol].find().limit(-1).sort('index', pm.DESCENDING).next()['columns']

                    cursor = self.tables[symbol].find({'index': {'$lt': end,
                                                        '$gte': start}}).sort('index', pm.DESCENDING)

                    data = [[item['data'][0][i] for i in range(len(item) - 2)] + [item['data'][0][2]] for item in cursor]

                    try:
                        obs = sample_trades(pd.DataFrame(data=data, columns=columns).set_index('trades_date_time',
                                                                    drop=True), "%dmin" % (freq))[-steps:]

                    except TypeError:
                        obs = False
                        df_steps += 1
                        continue

                    try:
                        assert obs.shape[0] == steps
                    except AssertionError:
                        obs = False
                        df_steps += 1
                        continue

                    try:
                        if hasattr(self.df, symbol):
                            if isinstance(self.df[symbol], pd.core.frame.DataFrame):
                                if self.online:
                                    # if hasattr(self.df, 'prev_position'):
                                    #     obs['prev_position'] = self.df.loc[obs.index, 'prev_position']
                                    if hasattr(self.df[symbol], 'position'):
                                        obs['position'] = self.df.loc[obs.index, (symbol, 'position')]
                                else:
                                    # Get position
                                    # if hasattr(self.df, 'prev_position'):
                                    #     obs['prev_position'] = self.df.loc[obs.index, 'prev_position']
                                    if hasattr(self.df[symbol], 'position'):
                                        obs['position'] = self.df.loc[obs.index, (symbol, 'position')]
                    except KeyError:
                        pass

                    obs = obs.astype(np.float64)
                    obs.ffill(inplace=True)
                    obs.fillna(1e-8, inplace=True)

                    try:
                        assert obs.shape[0] == steps
                    except AssertionError:
                        obs = False
                        continue

                assert isinstance(obs, pd.core.frame.DataFrame)

            return obs

        except Exception as e:
            self.logger.error(TradingEnvironment._get_obs, self.parse_error(e))
            return False

    def _get_order_type(self):
        return self.order_type

    def _get_fiat(self):
        assert self.online and self.fiat >= 0.0
        return self.fiat

    def _get_crypto(self, symbol):
        assert symbol in [s for s in self.crypto.keys()]
        assert self.online and self.crypto[symbol] >= 0.0
        return self.crypto[symbol]

    def _get_portval(self):
        try:
            assert self.online
            portval = convert_to.decimal('0.0')

            for symbol in self._get_df_symbols(no_fiat=True):
                val = self._get_crypto(symbol) * convert_to.decimal(self._get_obs(symbol, last_price=True))
                assert val >= 0.0
                portval += val

            portval += self._get_fiat()

            return portval
        except Exception as e:
            self.logger.error(TradingEnvironment._get_portval, self.parse_error(e))
            return False

    def _get_prev_portval(self):
        assert self.online and self.prev_val >= 0.0
        return self.prev_val

    def _get_posit(self, symbol):
        try:
            assert self.online
            assert self.posit[symbol] <= 1.0
            return self.posit[symbol]
        except Exception as e:
            self.logger.error(TradingEnvironment._get_posit, self.parse_error(e))
            return False

    def _get_prev_posit(self, symbol):
        assert self.online
        return self.prev_posit[symbol]

    # Exchange getter (make requests)
    def _get_exchange_fiat(self, symbol):
        try:
            assert self.online
            base, quote = symbol[:3], symbol[3:]
            balance = self.tapi.account_balance(base=base, quote=quote)
            return convert_to.decimal(balance[quote + '_balance'])

        except Exception as e:
            self.logger.error(TradingEnvironment._get_exchange_fiat, self.parse_error(e))
            return False

    def _get_exchange_crypto(self, symbol):
        try:
            assert self.online
            base, quote = symbol[:3], symbol[3:]
            balance = self.tapi.account_balance(base=base, quote=quote)
            return convert_to.decimal(balance[base + '_balance'])

        except Exception as e:
            self.logger.error(TradingEnvironment._get_exchange_crypto, self.parse_error(e))
            return False

    def _get_exchange_portval(self):
        # TODO: VALIDATE
        try:
            assert self.online

            portval = Decimal('0.0')
            for symbol in self._get_df_symbols(no_fiat=True):
                base, quote = symbol[:3], symbol[3:]
                ticker = self.tapi.ticker_hour(base=base, quote=quote)
                balance = self.tapi.account_balance(base=base, quote=quote)
                portval += convert_to.decimal(balance[base + '_balance']) * \
                           convert_to.decimal(ticker['last']) + \
                           convert_to.decimal(balance[quote + '_balance'])

            return portval
        except Exception as e:
            self.logger.error(TradingEnvironment._get_exchange_portval, self.parse_error(e))
            return False

    def _get_exchange_posit(self, symbol):
        # TODO: VALIDATE
        try:
            assert self.online

            base, quote = symbol[:3], symbol[3:]

            if symbol in self._get_df_symbols(no_fiat=True):
                balance = self.tapi.account_balance(base=base, quote=quote)
                ticker = self.tapi.ticker_hour(base=base, quote=quote)

                return  convert_to.decimal(balance[base + '_balance']) * \
                            convert_to.decimal(ticker['last']) / \
                            self._get_exchange_portval()

            elif symbol == 'fiat':

                balance = self.tapi.account_balance(base='btc', quote='usd')

                return  convert_to.decimal(balance['usd_balance']) / self._get_exchange_portval()

        except Exception as e:
            self.logger.error(TradingEnvironment._get_exchange_posit, self.parse_error(e))
            return False

    def _get_exchange_portifolio_posit(self):
        portifolio = []
        for symbol in self._get_df_symbols():
            portifolio.append(self._get_exchange_posit(symbol))
        return np.array(portifolio)

    def _get_balance(self, symbol):
        try:
            assert self.online
            base, quote = symbol[:3], symbol[3:]
            balance = self.tapi.account_balance(base=base, quote=quote)
            assert isinstance(balance, dict)
            return balance
        except Exception as e:
            self.logger.error(TradingEnvironment._get_balance, self.parse_error(e))
            return False

    def _rebalance_portifolio(self):
        pass # TODO _REBALANCE_PORTIFOLIO

    def _get_results(self):
        """
        Calculate online operation statistics
        :return:
        """
        self.results = self.df.copy()
        self.results['portval'] = convert_to.decimal(np.nan)
        self.results['benchmark'] = convert_to.decimal(np.nan)
        self.results['returns'] = convert_to.decimal(np.nan)
        self.results['benchmark_returns'] = convert_to.decimal(np.nan)
        self.results['alpha'] = convert_to.decimal(np.nan)
        self.results['beta'] = convert_to.decimal(np.nan)
        self.results['drawdown'] = convert_to.decimal(np.nan)
        self.results['sharpe'] = convert_to.decimal(np.nan)

        self.results['portval'] = self.df.crypto * self.df.close + self.df.fiat
        self.results['benchmark'] = self.df.close * self._get_init_fiat() / self.df.loc[self.session_begin_time].close - \
                                    self._get_tax() * self._get_init_fiat() / self.df.loc[self.session_begin_time].close
        self.results['returns'] = pd.to_numeric(self.results.portval).diff().fillna(1e-8)
        self.results['benchmark_returns'] = pd.to_numeric(self.results.benchmark).diff().fillna(1e-8)
        self.results['alpha'] = ec.utils.roll(self.results.returns,
                                              self.results.benchmark_returns,
                                              function=ec.alpha_aligned,
                                              risk_free=0.001,
                                              window=30)
        self.results['beta'] = ec.utils.roll(self.results.returns,
                                             self.results.benchmark_returns,
                                             function=ec.beta_aligned,
                                             window=30)
        self.results['drawdown'] = ec.roll_max_drawdown(self.results.returns, window=3)
        self.results['sharpe'] = ec.roll_sharpe_ratio(self.results.returns, window=3, risk_free=0.001)

    def _reset_status(self):
        self.status = {'OOD': False, 'Error': False, 'ValueError': False, 'ActionError': False}

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_email(self, email, psw):
        """
        Set Gmail address and password for log keeping
        :param email: str: Gmail address
        :param psw: str: account password
        :return:
        """
        try:
            assert isinstance(email, str) and isinstance(psw, str)
            self.email = email
            self.psw = psw
            self.logger.info(TradingEnvironment.set_email, "Email report address set to: %s" % (self.email))
        except Exception as e:
            self.logger.error(TradingEnvironment.set_email, self.parse_error(e))

    def _send_email(self, subject, body):
        try:
            assert isinstance(self.email, str) and isinstance(self.psw, str) and \
                   isinstance(subject, str) and isinstance(body, str)
            gmail_user = self.email
            gmail_pwd = self.psw
            FROM = self.email
            TO = self.email if type(self.email) is list else [self.email]
            SUBJECT = subject
            TEXT = body

            # Prepare actual message
            message = """From: %s\nTo: %s\nSubject: %s\n\n%s
                    """ % (FROM, ", ".join(TO), SUBJECT, TEXT)

            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.ehlo()
            server.starttls()
            server.login(gmail_user, gmail_pwd)
            server.sendmail(FROM, TO, message)
            server.close()

        except Exception as e:
            self.logger.error(TradingEnvironment._send_email, self.parse_error(e))
        finally:
            pass

    def parse_error(self, e):
        error_msg = '\n' + self.name + ' error -> ' + type(e).__name__ + ' in line ' + str(
            e.__traceback__.tb_lineno) + ': ' + str(e)
        return error_msg