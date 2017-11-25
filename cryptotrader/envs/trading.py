"""
Trading environment class
data: 12/10/2017
author: Tau
"""
from ..core import Env, ExchangeConnection, DataFeed
from ..spaces import *
from ..utils import Logger, safe_div, floor_datetime
from .utils import *

import os
import smtplib
from datetime import datetime, timedelta, timezone
from decimal import getcontext, localcontext, ROUND_UP, Decimal
from time import sleep, time
import pandas as pd
import empyrical as ec
from bokeh.layouts import column
from bokeh.palettes import inferno
from bokeh.plotting import figure, show
from bokeh.models import HoverTool, Legend

from ..exchange_api.poloniex import PoloniexError, Poloniex, RetryException
import json


# Decimal precision
getcontext().prec = 24

# Debug flag
debug = True

class BacktestDataFeed(DataFeed):
    """
    Data feeder for backtesting with TradingEnvironment.
    """
    # TODO WRITE TESTS
    def __init__(self, tapi, period, pairs=[], balance={}, load_dir=None):
        super().__init__(tapi, period, pairs, balance)
        self.ohlc_data = {}
        self.data_length = 0
        self.load_dir = load_dir

    def returnBalances(self):
        return self._balance

    def returnFeeInfo(self):
        return {'makerFee': '0.00150000',
                'nextTier': '600.00000000',
                'takerFee': '0.00250000',
                'thirtyDayVolume': '0.00000000'}

    def returnCurrencies(self):
        if self.load_dir:
            try:
                with open(self.load_dir+'/currencies.json') as file:
                    return json.load(file)
            except Exception as e:
                print(str(e.__cause__) + str(e))
                return self.tapi.returnCurrencies()
        else:
            return self.tapi.returnCurrencies()

    def download_data(self, start=None, end=None):
        # TODO WRITE TEST
        self.ohlc_data = {}
        self.data_length = None
        for pair in self.pairs:
            try:
                self.ohlc_data[pair] = pd.DataFrame.from_records(self.tapi.returnChartData(pair, period=self.period * 60,
                                                                   start=start, end=end
                                                                  ))
            except PoloniexError:
                try:
                    symbols = pair.split('_')
                    self.ohlc_data[pair] = self.pair_reciprocal(pd.DataFrame.from_records(
                        self.tapi.returnChartData(symbols[1] + '_' + symbols[0], period=self.period * 60,
                                                   start=start, end=end
                                                  )))
                except PoloniexError as e:
                    raise e



        for key in self.ohlc_data:
            if not self.data_length or self.ohlc_data[key].shape[0] < self.data_length:
                self.data_length = self.ohlc_data[key].shape[0]

        for key in self.ohlc_data:
            if self.ohlc_data[key].shape[0] != self.data_length:
                self.ohlc_data[key] = pd.DataFrame.from_records(self.tapi.returnChartData(key, period=self.period * 60,
                                                               start=self.ohlc_data[key].date.iloc[-self.data_length],
                                                               end=end
                                                               ))

            self.ohlc_data[key].set_index('date', inplace=True, drop=False)


        print("%d intervals, or %d days of data at %d minutes period downloaded." % (self.data_length, (self.data_length * self.period) /\
                                                                (24 * 60), self.period))

    def save_data(self, dir=None):
        """
        Save data to disk
        :param dir: str: directory relative to ./; eg './data/train
        :return:
        """
        for item in self.ohlc_data:
            self.ohlc_data[item].to_json(dir+'/'+str(item)+'_'+str(self.period)+'min', orient='records')

    def load_data(self, dir):
        """
        Load data form disk.
        JSON like data expected.
        :param dir: str: directory relative to self.load_dir; eg: './self.load_dir/dir'
        :return: None
        """
        self.ohlc_data = {}
        self.data_length = None
        for key in self.pairs:
            self.ohlc_data[key] = pd.read_json(self.load_dir + dir +'/'+str(key)+'_'+str(self.period)+'min', convert_dates=False,
                                                orient='records', date_unit='s', keep_default_dates=False, dtype=False)
            self.ohlc_data[key].set_index('date', inplace=True, drop=False)
            if not self.data_length:
                self.data_length = self.ohlc_data[key].shape[0]
            else:
                assert self.data_length == self.ohlc_data[key].shape[0]

    def returnChartData(self, currencyPair, period, start=None, end=None):
        try:
            data = json.loads(self.ohlc_data[currencyPair].loc[start:end, :].to_json(orient='records'))

            return data

        except json.JSONDecodeError:
            print("Bad exchange response.")

        except AssertionError as e:
            if "Invalid period" == e:
                raise PoloniexError("%d invalid candle period" % period)
            elif "Invalid pair" == e:
                raise PoloniexError("Invalid currency pair.")


class PaperTradingDataFeed(DataFeed):
    """
    Data feeder for paper trading with TradingEnvironment.
    """
    # TODO WRITE TESTS
    def __init__(self, tapi, period, pairs=[], balance={}):
        super().__init__(tapi, period, pairs, balance)

    def returnBalances(self):
        return self._balance

    def returnFeeInfo(self):
        return {'makerFee': '0.00150000',
                'nextTier': '600.00000000',
                'takerFee': '0.00250000',
                'thirtyDayVolume': '0.00000000'}


class PoloniexConnection(DataFeed):
    def __init__(self, tapi, period, pairs=[]):
        """
        :param tapi: exchange api instance: Exchange api instance
        :param period: int: Data period
        :param pairs: list: Pairs to trade
        """
        super().__init__(tapi, period, pairs)

    # Data feed methods
    @property
    def balance(self):
        return self.tapi.returnBalances()

    # Trade execution methods
    def sell(self, currencyPair, rate, amount, orderType=False):
        return self.tapi.sell(currencyPair, rate, amount, orderType=orderType)

    def buy(self, currencyPair, rate, amount, orderType=False):
        return self.tapi.buy(currencyPair, rate, amount, orderType=orderType)


class MultiExchangeConnection(DataFeed):
    """
    Exchanges connection helper.
    This class communicate with all exchanges apis.
    """
    def __init__(self, tapis={}, period=None, pairs=[]):
        """
        :param tapis: dict: Dictionary containing the exchanges names and api instances
        :param period: int: Data period
        :param pairs: list: Pairs to trade
        """
        super().__init__(tapis, period, pairs)


class TradingEnvironment(Env):
    """
    Trading environment base class
    """
    ## Setup methods
    def __init__(self, period, obs_steps, tapi, fiat="USDT", name="TradingEnvironment"):
        assert isinstance(name, str), "Name must be a string"
        self.name = name

        # Data feed api
        self.tapi = tapi

        # Environment configuration
        self.epsilon = convert_to.decimal('1E-8')
        self._obs_steps = None
        self._period = None
        self.pairs = []
        self._crypto = []
        self._fiat = None
        self.tax = {}

        # Dataframes
        self.obs_df = pd.DataFrame()
        self.portfolio_df = pd.DataFrame()
        self.action_df = pd.DataFrame()

        # Logging and debugging
        self.status = {'OOD': False,
                       'Error': False,
                       'ValueError': False,
                       'ActionError': False,
                       'NotEnoughFiat': False}

        if not os.path.exists('./logs'):
            os.makedirs('./logs')
        self.logger = Logger(self.name, './logs/')
        self.logger.info("Trading Environment initialization",
                         "Trading Environment Initialized!")

        # Setup
        self.period = period
        self.obs_steps = obs_steps

        # Init attributes for key sharing
        self.results = None
        self.action_space = None
        self.observation_space = None
        self.init_balance = None
        self._symbols = []

        self.add_pairs(self.tapi.pairs)
        self.fiat = fiat

    ## Env properties
    @property
    def obs_steps(self):
        return self._obs_steps

    @obs_steps.setter
    def obs_steps(self, value):
        assert isinstance(value, int), "Obs steps must be a integer."
        assert value >= 3, "Obs steps must be >= 3. Value: %s" % str(value)
        self._obs_steps = value

    @property
    def period(self):
        return self._period

    @period.setter
    def period(self, value):
        assert isinstance(value, int) and value >= 1,\
            "Period must be a integer >= 1."
        self._period = value

    @property
    def symbols(self):
        if self._symbols:
            return self._symbols
        else:
            symbols = []
            for pair in self.pairs:
                symbols.append(pair.split('_')[1])
            symbols.append(self._fiat)
            self._symbols = symbols
            return self._symbols

    @property
    def fiat(self):
        try:
            i = -1
            fiat = self.portfolio_df.get_value(self.portfolio_df.index[i], self._fiat)
            while not convert_to.decimal(fiat.is_finite()):
                i -= 1
                fiat = self.portfolio_df.get_value(self.portfolio_df.index[-i], self._fiat)
            return fiat
        except IndexError:
            self.logger.error(TradingEnvironment.crypto, "No valid value on portfolio dataframe.")
            raise KeyError
        except KeyError as e:
            self.logger.error(TradingEnvironment.fiat, "You must specify a fiat symbol first.")
            raise e
        except Exception as e:
            self.logger.error(TradingEnvironment.fiat, self.parse_error(e))
            raise e

    @fiat.setter
    def fiat(self, value):
        try:
            if isinstance(value, str):
                symbols = []
                for pair in self.pairs:
                    symbols.append(pair.split('_')[1])
                symbols.append(self.pairs[0].split('_')[0])
                assert value in symbols, "Fiat not in symbols."
                self._fiat = value
                symbols.remove(self._fiat)
                self._crypto = symbols

            elif isinstance(value, Decimal) or isinstance(value, float) or isinstance(value, int):
                self.portfolio_df.at[self.timestamp, self._fiat] = convert_to.decimal(value)

            elif isinstance(value, dict):
                try:
                    timestamp = value['timestamp']
                except KeyError:
                    timestamp = self.timestamp
                self.portfolio_df.at[timestamp, self._fiat] = convert_to.decimal(value[self._fiat])

        except IndexError:
            raise AssertionError('You must enter pairs before set fiat.')

        except Exception as e:
            self.logger.error(TradingEnvironment.fiat, self.parse_error(e))
            raise e

    @property
    def crypto(self):
        try:
            crypto = {}
            for symbol in self._crypto:
                crypto[symbol] = self.get_crypto(symbol)
            return crypto
        except KeyError as e:
            self.logger.error(TradingEnvironment.crypto, "No valid value on portfolio dataframe.")
            raise e
        except Exception as e:
            self.logger.error(TradingEnvironment.crypto, self.parse_error(e))
            raise e

    def get_crypto(self, symbol):
        try:
            i = -1
            value = self.portfolio_df.get_value(self.portfolio_df.index[i], symbol)
            while not convert_to.decimal(value).is_finite():
                i -= 1
                value = self.portfolio_df.get_value(self.portfolio_df.index[i], symbol)
            return value

        except IndexError:
            self.logger.error(TradingEnvironment.crypto, "No valid value on portfolio dataframe.")
            raise KeyError
        except KeyError as e:
            self.logger.error(TradingEnvironment.crypto, "No valid value on portfolio dataframe.")
            raise e
        except Exception as e:
            self.logger.error(TradingEnvironment.crypto, self.parse_error(e))
            raise e

    @crypto.setter
    def crypto(self, values):
        try:
            # assert isinstance(values, dict), "Crypto value must be a dictionary containing the currencies balance."
            try:
                timestamp = values['timestamp']
            except KeyError:
                timestamp = self.timestamp
            for symbol, value in values.items():
                if symbol not in [self._fiat, 'timestamp']:
                    self.portfolio_df.at[timestamp, symbol] = convert_to.decimal(value)

        except TypeError:
            raise AssertionError("Crypto value must be a dictionary containing the currencies balance.")

        except Exception as e:
            self.logger.error(TradingEnvironment.crypto, self.parse_error(e))
            raise e

    @property
    def balance(self):
        # return self.portfolio_df.ffill().loc[self.portfolio_df.index[-1], self.symbols].to_dict()
        balance = self.crypto
        balance.update({self._fiat: self.fiat})
        return balance

    @balance.setter
    def balance(self, values):
        try:
            assert isinstance(values, dict), "Balance must be a dictionary containing the currencies amount."
            try:
                timestamp = values['timestamp']
            except KeyError:
                timestamp = self.timestamp
            for symbol, value in values.items():
                if symbol is not 'timestamp':
                    self.portfolio_df.at[timestamp, symbol] = convert_to.decimal(value)

        except Exception as e:
            self.logger.error(TradingEnvironment.balance, self.parse_error(e))
            raise e

    @property
    def portval(self):
        return self.calc_total_portval()

    @portval.setter
    def portval(self, value):
        try:
            self.portfolio_df.at[value['timestamp'], 'portval'] = convert_to.decimal(value['portval'])
        except KeyError:
            self.portfolio_df.at[self.timestamp, 'portval'] = convert_to.decimal(value['portval'])
        except TypeError:
            self.portfolio_df.at[self.timestamp, 'portval'] = convert_to.decimal(value)

        except Exception as e:
            self.logger.error(TradingEnvironment.portval, self.parse_error(e))
            raise e

    def add_pairs(self, *args):

        universe = self.tapi.returnCurrencies()

        for arg in args:
            if isinstance(arg, str):
                if set(arg.split('_')).issubset(universe):
                    self.pairs.append(arg)
                else:
                    self.logger.error(TradingEnvironment.add_pairs, "Symbol not found on exchange currencies.")

            elif isinstance(arg, list):
                for item in arg:
                    if set(item.split('_')).issubset(universe):
                        if isinstance(item, str):
                            self.pairs.append(item)
                        else:
                            self.logger.error(TradingEnvironment.add_pairs, "Symbol name must be a string")

            else:
                self.logger.error(TradingEnvironment.add_pairs, "Symbol name must be a string")

    ## Data feed methods
    @property
    def timestamp(self):
        # return floor_datetime(datetime.now(timezone.utc) - timedelta(minutes=self.period), self.period)
        # Poloniex returns utc timestamp delayed from one bar
        return datetime.now(timezone.utc) - timedelta(minutes=self.period)

    # Exchange data getters
    def get_balance(self):
        try:
            balance = self.tapi.returnBalances()

            filtered_balance = {}
            for symbol in self.symbols:
                filtered_balance[symbol] = balance[symbol]

            return filtered_balance

        except Exception as e:
            self.logger.error(TradingEnvironment.get_balance, self.parse_error(e))
            raise e

    def get_fee(self, symbol, fee_type='takerFee'):
        # TODO MAKE IT UNIVERSAL
        try:
            fees = self.tapi.returnFeeInfo()

            assert fee_type in ['takerFee', 'makerFee'], "fee_type must be whether 'takerFee' or 'makerFee'."
            return convert_to.decimal(fees[fee_type])

        except Exception as e:
            self.logger.error(TradingEnvironment.get_fee, self.parse_error(e))
            raise e

    # High frequency getter
    def get_pair_trades(self, pair, start=None, end=None):
        # TODO WRITE TEST
        # TODO FINISH THIS
        try:
            # Pool data from exchage
            if isinstance(end, float):
                data = self.tapi.marketTradeHist(pair, end=end)
            else:
                data = self.tapi.marketTradeHist(pair)
            df = pd.DataFrame.from_records(data)

            # Get more data from exchange until have enough to make obs_steps rows
            if isinstance(start, float):
                while datetime.fromtimestamp(start) < \
                        datetime.strptime(df.date.iat[-1], "%Y-%m-%d %H:%M:%S"):

                    market_data = self.tapi.marketTradeHist(pair, end=datetime.timestamp(
                        datetime.strptime(df.date.iat[-1], "%Y-%m-%d %H:%M:%S")))

                    df2 = pd.DataFrame.from_records(market_data).set_index('globalTradeID')
                    appended = False
                    i = 0
                    while not appended:
                        try:
                            df = df.append(df2.iloc[i:], verify_integrity=True)
                            appended = True
                        except ValueError:
                            i += 1

            else:
                while datetime.strptime(df.date.iat[0], "%Y-%m-%d %H:%M:%S") - \
                        timedelta(minutes=self.period * self.obs_steps) < \
                        datetime.strptime(df.date.iat[-1], "%Y-%m-%d %H:%M:%S"):

                    market_data = self.tapi.marketTradeHist(pair, end=datetime.timestamp(
                        datetime.strptime(df.date.iat[-1], "%Y-%m-%d %H:%M:%S")))

                    df2 = pd.DataFrame.from_records(market_data).set_index('globalTradeID')
                    appended = False
                    i = 0
                    while not appended:
                        try:
                            df = df.append(df2.iloc[i:], verify_integrity=True)
                            appended = True
                        except ValueError:
                            i += 1

                return df

        except Exception as e:
            self.logger.error(TradingEnvironment.get_pair_trades, self.parse_error(e))
            raise e

    def sample_trades(self, pair, start=None, end=None):
        # TODO WRITE TEST
        df = self.get_pair_trades(pair, start=start, end=end)

        period = "%dmin" % self.period

        # Sample the trades into OHLC data
        df['rate'] = df['rate'].ffill().apply(convert_to.decimal, raw=True)
        df['amount'] = df['amount'].apply(convert_to.decimal, raw=True)
        df.index = df.date.apply(pd.to_datetime, raw=True)

        # TODO REMOVE NANS
        index = df.resample(period).first().index
        out = pd.DataFrame(index=index)

        out['open'] = convert_and_clean(df['rate'].resample(period).first())
        out['high'] = convert_and_clean(df['rate'].resample(period).max())
        out['low'] = convert_and_clean(df['rate'].resample(period).min())
        out['close'] = convert_and_clean(df['rate'].resample(period).last())
        out['volume'] = convert_and_clean(df['amount'].resample(period).sum())

        return out

    # Low frequency getter
    def get_ohlc(self, symbol, index):
        # Get range
        start = index[0]
        end = index[-1]

        # Call for data
        ohlc_df = pd.DataFrame.from_records(self.tapi.returnChartData(symbol,
                                                                        period=self.period * 60,
                                                                        start=datetime.timestamp(start),
                                                                        end=datetime.timestamp(end)),
                                                                        nrows=index.shape[0])
        # TODO 1 FIND A BETTER WAY
        # TODO: FIX TIMESTAMP

        # Set index
        ohlc_df.set_index(ohlc_df.date.transform(lambda x: datetime.fromtimestamp(x).astimezone(timezone.utc)),
                          inplace=True, drop=True)

        # Get right values to fill nans
        # TODO: FIND A BETTER PERFORMANCE METHOD
        last_close = ohlc_df.get_value(ohlc_df.close.last_valid_index(), 'close')
        fill_dict = {col: last_close for col in ['open', 'high', 'low', 'close']}
        fill_dict.update({'volume': '0E-8'})
        # Reindex with desired time range and fill nans
        ohlc_df = ohlc_df[['open','high','low','close',
                           'volume']].reindex(index).asfreq("%dT" % self.period).fillna(fill_dict)

        return ohlc_df.astype(str)#.fillna('0.0')

    # Observation maker
    def get_history(self, start=None, end=None, portfolio_vector=False):
        try:
            obs_list = []
            keys = []

            # Make desired index
            is_bounded = True
            if not end:
                end = self.timestamp
                is_bounded = False
            if not start:
                start = end - timedelta(minutes=self.period * self.obs_steps)
                index = pd.date_range(start=start,
                                      end=end,
                                      freq="%dT" % self.period).ceil("%dT" % self.period)[-self.obs_steps:]
                is_bounded = False
            else:
                index = pd.date_range(start=start,
                                      end=end,
                                      freq="%dT" % self.period).ceil("%dT" % self.period)

            if portfolio_vector:
                # Get portfolio observation
                port_vec = self.get_sampled_portfolio(index)
                if port_vec.shape[0] == 0:
                    port_vec = self.get_sampled_portfolio().iloc[-1:]
                    port_vec.index = [index[0]]

                # Get pairs history
                for pair in self.pairs:
                    keys.append(pair)
                    history = self.get_ohlc(pair, index)

                    history = pd.concat([history, port_vec[pair.split('_')[1]]], axis=1)
                    obs_list.append(history)

                # Get fiat history
                keys.append(self._fiat)
                obs_list.append(port_vec[self._fiat])

                # Concatenate dataframes
                obs = pd.concat(obs_list, keys=keys, axis=1)

                # Fill missing portfolio observations
                cols_to_bfill = [col for col in zip(self.pairs, self.symbols)] + [(self._fiat, self._fiat)]
                obs = obs.fillna(obs[cols_to_bfill].ffill().bfill())

                if not is_bounded:
                    assert obs.shape[0] >= self.obs_steps, "Dataframe is to small. Shape: %s" % str(obs.shape)

                return obs.apply(convert_to.decimal, raw=True)
            else:
                # Get history
                for pair in self.pairs:
                    keys.append(pair)
                    history = self.get_ohlc(pair, index)
                    obs_list.append(history)

                # Concatenate
                obs = pd.concat(obs_list, keys=keys, axis=1)

                # Check size
                if not is_bounded:
                    assert obs.shape[0] >= self.obs_steps, "Dataframe is to small. Shape: %s" % str(obs.shape)

                return obs.apply(convert_to.decimal, raw=True)

        except Exception as e:
            self.logger.error(TradingEnvironment.get_history, self.parse_error(e))
            raise e

    def get_observation(self, portfolio_vector=False):
        try:
            self.obs_df = self.get_history(portfolio_vector=portfolio_vector)
            return self.obs_df

        # except PoloniexError:
        #     sleep(1)
        #     self.obs_df = self.get_history(portfolio_vector=portfolio_vector)
        #     return self.obs_df

        except Exception as e:
            self.logger.error(TradingEnvironment.get_observation, self.parse_error(e))
            raise e

    def get_sampled_portfolio(self, index=None):
        if index is None:
            start = self.portfolio_df.index[0]
            end = self.portfolio_df.index[-1]

        else:
            start = index[0]
            end = index[-1]

        # TODO 1 FIND A BETTER WAY
        if start != end:
            return self.portfolio_df.loc[start:end].resample("%dmin" % self.period).last()
        else:
            return self.portfolio_df.loc[:end].resample("%dmin" % self.period).last()

    def get_sampled_actions(self, index=None):
        if index is None:
            start = self.portfolio_df.index[0]
            end = self.portfolio_df.index[-1]

        else:
            start = index[0]
            end = index[-1]

        # TODO 1 FIND A BETTER WAY
        return self.action_df.loc[start:end].resample("%dmin" % self.period).last()

    ## Trading methods
    def get_open_price(self, symbol, timestamp=None):
        if not timestamp:
            timestamp = self.obs_df.index[-1]
        return self.obs_df.get_value(timestamp, ("%s_%s" % (self._fiat, symbol), 'open'))

    def calc_total_portval(self, timestamp=None):
        portval = convert_to.decimal('0E-8')

        for symbol in self._crypto:
            portval = self.get_crypto(symbol).fma(self.get_open_price(symbol, timestamp), portval)
        portval += self.fiat

        return portval

    def calc_posit(self, symbol, portval):
        """
        Calculate current position vector
        :param symbol: str: Symbol name
        :param portval: Decimal: Portfolio value
        :return:
        """
        if symbol == self._fiat:
            return safe_div(self.fiat, portval)
        else:
            return safe_div(self.get_crypto(symbol) * self.get_open_price(symbol), portval)

    def calc_portfolio_vector(self):
        portfolio = np.empty(len(self.symbols), dtype=Decimal)
        portval = self.calc_total_portval()
        for i, symbol in enumerate(self.symbols):
            portfolio[i] = self.calc_posit(symbol, portval)
        return portfolio

    def assert_action(self, action):
        # TODO WRITE TEST
        try:
            action = convert_to.decimal(action)
            # normalize
            if action.sum() != convert_to.decimal('1.00000000'):
                action = safe_div(action, action.sum())

            action[-1] += convert_to.decimal('1.00000000') - action.sum()

            assert action.sum() - convert_to.decimal('1.00000000') < convert_to.decimal('1e-8')
            return action

        except AssertionError:
            action = safe_div(action, action.sum())
            action[-1] += convert_to.decimal('1.00000000') - action.sum()
            try:
                assert action.sum() - convert_to.decimal('1.00000000') < convert_to.decimal('1e-8')
                return action
            except AssertionError:
                action = safe_div(action, action.sum())
                action[-1] += convert_to.decimal('1.00000000') - action.sum()
                assert action.sum() - convert_to.decimal('1.00000000') < convert_to.decimal('1e-8')
                return action

        except Exception as e:
            self.logger.error(TradingEnvironment.assert_action, self.parse_error(e))
            raise e

    def log_action(self, timestamp, symbol, value):
        if symbol == 'online':
            self.action_df.at[timestamp, symbol] = value
        else:
            self.action_df.at[timestamp, symbol] = convert_to.decimal(value)

    def log_action_vector(self, timestamp, vector, online):
        for i, symbol in enumerate(self.symbols):
            self.log_action(timestamp, symbol, vector[i])
        self.log_action(timestamp, 'online', online)

    def get_last_portval(self):
        try:
            i = -1
            portval = self.portfolio_df.get_value(self.portfolio_df.index[i], 'portval')
            while not convert_to.decimal(portval).is_finite():
                i -= 1
                portval = self.portfolio_df.get_value(self.portfolio_df.index[i], 'portval')

            return portval
        except Exception as e:
            self.logger.error(TradingEnvironment.get_last_portval, self.parse_error(e))
            raise e

    def get_reward(self):
        """
        Regret loss function

        Reference:
            A.Agarwal, E.Hazan, S.Kale, R.E.Schapire.
            Algorithms for Portfolio Management based on the Newton Method, 2006.
            http://machinelearning.wustl.edu/mlpapers/paper_files/icml2006_AgarwalHKS06.pdf

        :return: numpy float:
        """
        # TODO TEST

        # Regret Calculation
        price = self.obs_df.xs('open', level=1, axis=1).iloc[-2:].astype('f').values
        price_change = np.append(price[-1] / (price[-2] + 1e-8), [1.0])
        constrained_price_change = price_change / price_change.max()

        port_log_return = np.log(np.dot(self.action_df.iloc[-1].astype('f').values[:-1], constrained_price_change))
        n_pairs = len(self.pairs)
        bench_action_vec = np.append(np.ones(n_pairs, dtype='f') / n_pairs, [0.0])

        bench_log_return = np.log(np.dot(bench_action_vec, constrained_price_change))

        #
        # port_return = np.log(safe_div(self.portval, self.get_last_portval()))
        # bench_return = np.log(safe_div(self.portval, self.get_last_portval()))

        return port_log_return - bench_log_return

    def simulate_trade(self, action, timestamp):
        """
        Simulates trade on exchange environment
        :param action: np.array: Desired portfolio vector
        :param timestamp: datetime.datetime: Trade time
        :return: None
        """
        # TODO: IMPLEMENT SLIPPAGE MODEL
        try:
            # Assert inputs
            action = self.assert_action(action)
            # for symbol in self._get_df_symbols(no_fiat=True): TODO FIX THIS
            #     self.observation_space.contains(observation[symbol])
            # assert isinstance(timestamp, pd.Timestamp)

            # Log desired action
            self.log_action_vector(timestamp, action, False)

            # Calculate position change given action
            posit_change = (action - self.calc_portfolio_vector())[:-1]

            # Get initial portval
            portval = self.calc_total_portval()

            # Sell assets first
            for i, change in enumerate(posit_change):
                if change < convert_to.decimal('0E-8'):

                    symbol = self.symbols[i]

                    crypto_pool = safe_div(portval * action[i], self.get_open_price(symbol))

                    with localcontext() as ctx:
                        ctx.rounding = ROUND_UP

                        fee = portval * change.copy_abs() * self.tax[symbol]

                    self.fiat = {self._fiat: self.fiat + portval.fma(change.copy_abs(), -fee), 'timestamp': timestamp}

                    self.crypto = {symbol: crypto_pool, 'timestamp': timestamp}

            # Uodate prev portval with deduced taxes
            portval = self.calc_total_portval()

            # Then buy some goods
            for i, change in enumerate(posit_change):
                if change > convert_to.decimal('0E-8'):

                    symbol = self.symbols[i]

                    self.fiat = {self._fiat: self.fiat - portval * change.copy_abs(), 'timestamp': timestamp}

                    # if fiat_pool is negative, deduce it from portval and clip
                    if self.fiat < convert_to.decimal('0E-8'):
                        portval += self.fiat
                        self.fiat = {self._fiat: convert_to.decimal('0E-8'), 'timestamp': timestamp}

                    with localcontext() as ctx:
                        ctx.rounding = ROUND_UP

                        fee = self.tax[symbol] * portval * change

                    crypto_pool = safe_div(portval.fma(action[i], -fee), self.get_open_price(symbol))

                    self.crypto = {symbol: crypto_pool, 'timestamp': timestamp}

            # Log executed action and final balance
            self.log_action_vector(self.timestamp, self.calc_portfolio_vector(), True)

            # Update portfolio_df
            final_balance = self.balance
            final_balance['timestamp'] = timestamp
            self.balance = final_balance

            # Calculate new portval
            self.portval = {'portval': self.calc_total_portval(),
                            'timestamp': self.portfolio_df.index[-1]}

        except Exception as e:
            self.logger.error(TradingEnvironment.simulate_trade, self.parse_error(e))
            if hasattr(self, 'email'):
                self.send_email("TradingEnvironment Error: %s at %s" % (e,
                                datetime.strftime(self.timestamp, "%Y-%m-%d %H:%M:%S")),
                                self.parse_error(e))
            raise e

    ## Env methods
    def set_observation_space(self):
        # Observation space:
        obs_space = []
        # OPEN, HIGH, LOW, CLOSE
        for _ in range(4):
            obs_space.append(Box(0.0, 1e12, 1))
        # VOLUME
        obs_space.append(Box(0.0, 1e12, 1))
        # POSITION
        obs_space.append(Box(0.0, 1.0, 1))

        self.observation_space = Tuple(obs_space)

    def set_action_space(self):
        # Action space
        self.action_space = Box(0., 1., len(self.symbols))
        # self.logger.info(TrainingEnvironment.set_action_space, "Setting environment with %d symbols." % (len(self.symbols)))

    def reset_status(self):
        self.status = {'OOD': False, 'Error': False, 'ValueError': False, 'ActionError': False,
                       'NotEnoughFiat': False}

    def reset(self):
        """
        Setup env with initial values
        :return: pandas DataFrame: observation
        """
        raise NotImplementedError()

    ## Analytics methods
    def get_results(self, window=3):
        """
        Calculate arbiter desired actions statistics
        :return:
        """
        self.results = self.get_sampled_portfolio().join(self.get_sampled_actions(), rsuffix='_posit')[1:].ffill()

        obs = self.get_history(self.results.index[0], self.results.index[-1])

        self.results['benchmark'] = convert_to.decimal('0E-8')
        self.results['returns'] = convert_to.decimal(np.nan)
        self.results['benchmark_returns'] = convert_to.decimal(np.nan)
        self.results['alpha'] = convert_to.decimal(np.nan)
        self.results['beta'] = convert_to.decimal(np.nan)
        self.results['drawdown'] = convert_to.decimal(np.nan)
        self.results['sharpe'] = convert_to.decimal(np.nan)

        ## Calculate benchmark portifolio, just equaly distribute money over all the assets
        # Calc init portval
        init_portval = Decimal('0E-8')
        init_time = self.results.index[0]
        for symbol in self._crypto:
            init_portval += convert_to.decimal(self.init_balance[symbol]) * \
                           obs.get_value(init_time, (self._fiat + '_' + symbol, 'open'))
        init_portval += convert_to.decimal(self.init_balance[self._fiat])

        with localcontext() as ctx:
            ctx.rounding = ROUND_UP
            for symbol in self.pairs:
                self.results[symbol+'_benchmark'] = (Decimal('1') - self.tax[symbol.split('_')[1]]) * obs[symbol, 'open'] * \
                                            init_portval / (obs.get_value(init_time,
                                            (symbol, 'open')) * Decimal(self.action_space.low.shape[0] - 1))
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
        self.results['drawdown'] = ec.roll_max_drawdown(self.results.returns, window=int(window))
        self.results['sharpe'] = ec.roll_sharpe_ratio(self.results.returns, window=int(window + 5), risk_free=0.001)

        return self.results

    def plot_results(self):
        def config_fig(fig):
            fig.background_fill_color = "black"
            fig.background_fill_alpha = 0.1
            fig.border_fill_color = "#232323"
            fig.outline_line_color = "#232323"
            fig.title.text_color = "whitesmoke"
            fig.xaxis.axis_label_text_color = "whitesmoke"
            fig.yaxis.axis_label_text_color = "whitesmoke"
            fig.yaxis.major_label_text_color = "whitesmoke"
            fig.xaxis.major_label_orientation = np.pi / 4
            fig.grid.grid_line_alpha = 0.1
            fig.grid.grid_line_dash = [6, 4]

        df = self.get_results().astype(np.float64)

        # Results figures
        results = {}

        # Position
        pos_hover = HoverTool(
            tooltips=[
                ('date', '<span style="color: #000000;">@x{%F, %H:%M}</span>'),
                ('position', '<span style="color: #000000;">@y{%f}</span>'),
                ],

            formatters={
                'x': 'datetime',  # use 'datetime' formatter for 'date' field
                'y': 'printf',  # use 'printf' formatter for 'adj close' field
                },

            # display a tooltip whenever the cursor is vertically in line with a glyph
            mode='vline'
            )

        p_pos = figure(title="Position over time",
                       x_axis_type="datetime",
                       x_axis_label='timestep',
                       y_axis_label='position',
                       plot_width=900, plot_height=300,
                       tools=['crosshair','reset','xwheel_zoom','pan,box_zoom', pos_hover],
                       toolbar_location="above"
                       )
        config_fig(p_pos)

        palettes = inferno(len(self.symbols))
        legend = []
        for i, symbol in enumerate(self.symbols):
            results[symbol + '_posit'] = p_pos.line(df.index, df[symbol + '_posit'], color=palettes[i], line_width=1.2)#, muted_color=palettes[i], muted_alpha=0.2)
            p_pos.legend.click_policy = "hide"
            legend.append((str(symbol), [results[symbol + '_posit']]))

        p_pos.add_layout(Legend(items=legend, location=(0, -31)), 'right')
        p_pos.legend.click_policy = "hide"
        # Portifolio and benchmark values
        val_hover = HoverTool(
            tooltips=[
                ('date', '<span style="color: #000000;">@x{%F, %H:%M}</span>'),
                ('val', '<span style="color: #000000;">$@y{%0.2f}</span>'),
                ],

            formatters={
                'x': 'datetime',  # use 'datetime' formatter for 'date' field
                'y': 'printf',  # use 'printf' formatter for 'adj close' field
                },

            # display a tooltip whenever the cursor is vertically in line with a glyph
            mode='vline'
            )

        p_val = figure(title="Portifolio / Benchmark Value",
                       x_axis_type="datetime",
                       x_axis_label='timestep',
                       y_axis_label='value',
                       plot_width=900, plot_height=400,
                       tools=['crosshair', 'reset', 'xwheel_zoom', 'pan,box_zoom', val_hover],
                       toolbar_location="above"
                       )
        config_fig(p_val)

        results['portval'] = p_val.line(df.index, df.portval, color='green', line_width=1.2)
        results['benchmark'] = p_val.line(df.index, df.benchmark, color='red', line_width=1.2)

        p_val.add_layout(Legend(items=[("portval", [results['portval']]),
                                       ("benchmark", [results['benchmark']])], location=(0, -31)), 'right')
        p_val.legend.click_policy = "hide"

        # Individual assets portval
        p_pval = figure(title="Pair Performance",
                       x_axis_type="datetime",
                       x_axis_label='timestep',
                       y_axis_label='position',
                       plot_width=900, plot_height=400,
                       tools=['crosshair', 'reset', 'xwheel_zoom', 'pan,box_zoom', val_hover],
                       toolbar_location="above"
                       )
        config_fig(p_pval)

        legend = []
        for i, symbol in enumerate(self.pairs):
            results[symbol+'_benchmark'] = p_pval.line(df.index, df[symbol+'_benchmark'], color=palettes[i], line_width=1.2)
            legend.append((symbol,[results[symbol+'_benchmark']]))


        p_pval.add_layout(Legend(items=legend, location=(0, -31)), 'right')
        p_pval.legend.click_policy = "hide"

        # Portifolio and benchmark returns
        p_ret = figure(title="Portifolio / Benchmark Returns",
                       x_axis_type="datetime",
                       x_axis_label='timestep',
                       y_axis_label='Returns',
                       plot_width=900, plot_height=200,
                       tools=['crosshair','reset','xwheel_zoom','pan,box_zoom'],
                       toolbar_location="above"
                       )
        config_fig(p_ret)

        results['bench_ret'] = p_ret.line(df.index, df.benchmark_returns, color='red', line_width=1.2)
        results['port_ret'] = p_ret.line(df.index, df.returns, color='green', line_width=1.2)

        p_ret.add_layout(Legend(items=[("bench returns", [results['bench_ret']]),
                                       ("port returns", [results['port_ret']])], location=(0, -31)), 'right')
        p_ret.legend.click_policy = "hide"

        p_hist = figure(title="Portifolio Value Pct Change Distribution",
                        x_axis_label='Pct Change',
                        y_axis_label='frequency',
                        plot_width=900, plot_height=300,
                        tools='crosshair,reset,xwheel_zoom,pan,box_zoom',
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
                         plot_width=900, plot_height=200,
                         tools='crosshair,reset,xwheel_zoom,pan,box_zoom',
                         toolbar_location="above"
                         )
        config_fig(p_alpha)

        results['alpha'] = p_alpha.line(df.index, df.alpha, color='yellow', line_width=1.2)

        # Portifolio rolling beta
        p_beta = figure(title="Portifolio rolling beta",
                        x_axis_type="datetime",
                        x_axis_label='timestep',
                        y_axis_label='beta',
                        plot_width=900, plot_height=200,
                        tools='crosshair,reset,xwheel_zoom,pan,box_zoom',
                        toolbar_location="above"
                        )
        config_fig(p_beta)

        results['beta'] = p_beta.line(df.index, df.beta, color='yellow', line_width=1.2)

        # Rolling Drawdown
        p_dd = figure(title="Portifolio rolling drawdown",
                      x_axis_type="datetime",
                      x_axis_label='timestep',
                      y_axis_label='drawdown',
                      plot_width=900, plot_height=200,
                      tools='crosshair,reset,xwheel_zoom,pan,box_zoom',
                      toolbar_location="above"
                      )
        config_fig(p_dd)

        results['drawdown'] = p_dd.line(df.index, df.drawdown, color='red', line_width=1.2)

        # Portifolio Sharpe ratio
        p_sharpe = figure(title="Portifolio rolling Sharpe ratio",
                          x_axis_type="datetime",
                          x_axis_label='timestep',
                          y_axis_label='Sharpe ratio',
                          plot_width=900, plot_height=200,
                          tools='crosshair,reset,xwheel_zoom,pan,box_zoom',
                          toolbar_location="above"
                          )
        config_fig(p_sharpe)

        results['sharpe'] = p_sharpe.line(df.index, df.sharpe, color='yellow', line_width=1.2)

        print("\n################### > Portifolio Performance Analysis < ###################\n")
        print("Portifolio excess Sharpe:                 %f" % ec.excess_sharpe(df.returns, df.benchmark_returns))
        print("Portifolio / Benchmark Sharpe ratio:      %f / %f" % (ec.sharpe_ratio(df.returns),
                                                                     ec.sharpe_ratio(df.benchmark_returns)))
        print("Portifolio / Benchmark Omega ratio:       %f / %f" % (ec.omega_ratio(df.returns),
                                                                     ec.omega_ratio(df.benchmark_returns)))
        print("Portifolio / Benchmark max drawdown:      %f / %f" % (ec.max_drawdown(df.returns),
                                                                     ec.max_drawdown(df.benchmark_returns)))

        results['handle'] = show(column(p_val, p_pval, p_pos, p_ret, p_hist, p_sharpe, p_dd, p_alpha, p_beta),
                                 notebook_handle=True)

        return results

    ## Helper methods
    def parse_error(self, e):
        error_msg = '\n' + self.name + ' error -> ' + type(e).__name__ + ' in line ' + str(
            e.__traceback__.tb_lineno) + ': ' + str(e)
        return error_msg

    def set_email(self, email):
        """
        Set Gmail address and password for log keeping
        :param email: str: Gmail address
        :param psw: str: account password
        :return:
        """
        try:
            assert isinstance(email, dict)
            self.email = email
            self.logger.info(TradingEnvironment.set_email, "Email report address set to: %s" % (str([email[key] for key in email if key == 'to'])))
        except Exception as e:
            self.logger.error(TradingEnvironment.set_email, self.parse_error(e))

    def send_email(self, subject, body):
        try:
            assert isinstance(self.email, dict) and \
                   isinstance(subject, str) and isinstance(body, str)
            for key in self.email:
                if key == 'email':
                    gmail_user = self.email[key]
                elif key == 'psw':
                    gmail_pwd = self.email[key]
                elif key == 'to':
                    TO = self.email[key] if type(self.email[key]) is list else [self.email[key]]

            FROM = gmail_user
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
            self.logger.error(TradingEnvironment.send_email, self.parse_error(e))
        finally:
            pass


class BacktestEnvironment(TradingEnvironment):
    """
    Backtest environment for financial strategies history testing
    """
    def __init__(self, period, obs_steps, tapi, fiat, name):
        assert isinstance(tapi, BacktestDataFeed), "Backtest tapi must be a instance of BacktestDataFeed."
        self.index = obs_steps
        super().__init__(period, obs_steps, tapi, fiat, name)
        self.data_length = None
        self.training = False

    @property
    def timestamp(self):
        return datetime.fromtimestamp(self.tapi.ohlc_data[self.tapi.pairs[0]].index[self.index]).astimezone(timezone.utc)

    def get_ohlc(self, symbol, index):
        # Get range
        start = index[0]
        end = index[-1]

        # Call for data
        ohlc_df = pd.DataFrame.from_records(self.tapi.returnChartData(symbol,
                                                                        period=self.period * 60,
                                                                        start=datetime.timestamp(start),
                                                                        end=datetime.timestamp(end)),
                                                                        nrows=index.shape[0])
        # TODO 1 FIND A BETTER WAY
        # TODO: FIX TIMESTAMP
        # Set index

        ohlc_df.set_index(ohlc_df.date.transform(lambda x: datetime.fromtimestamp(x).astimezone(timezone.utc)),
                          inplace=True, drop=True)

        # Disabled fill on backtest for performance.
        # We assume that backtest data feed will not return nan values

        # Get right values to fill nans
        # fill_dict = {col: ohlc_df.loc[ohlc_df.close.last_valid_index(), 'close'] for col in ['open', 'high', 'low', 'close']}
        # fill_dict.update({'volume': '0E-8'})
        # Reindex with desired time range and fill nans
        ohlc_df = ohlc_df[['open','high','low','close',
                           'volume']].reindex(index).asfreq("%dT" % self.period)#.fillna(fill_dict)

        return ohlc_df.astype(str)

    def reset(self, reset_dfs=True):
        """
        Setup env with initial values
        :param reset_dfs: bool: Reset log dfs
        :return: pandas DataFrame: Initial observation
        """
        try:
            # Reset index
            self.data_length = self.tapi.data_length

            if self.training:
                self.index = np.random.random_integers(self.obs_steps, self.data_length - 3)
            else:
                self.index = self.obs_steps

            # Reset log dfs
            if reset_dfs:
                self.obs_df = pd.DataFrame()
                self.portfolio_df = pd.DataFrame()

            # Set spaces
            self.set_observation_space()
            self.set_action_space()

            # Reset balance
            self.balance = self.init_balance = self.get_balance()

            # Get fee values
            for symbol in self.symbols:
                self.tax[symbol] = convert_to.decimal(self.get_fee(symbol))

            # Get new index
            self.index += 1

            obs = self.get_observation(True)

            if reset_dfs:
                self.action_df = pd.DataFrame([list(self.calc_portfolio_vector()) + [False]],
                                              columns=list(self.symbols) + ['online'],
                                              index=[self.timestamp])

            # Reset portfolio value
            self.portval = {'portval': self.calc_total_portval(self.obs_df.index[-1]),
                            'timestamp': self.portfolio_df.index[-1]}

            # Return first observation
            return obs.astype(np.float64)

        except IndexError:
            print("Insufficient tapi data. You must choose a bigger time span or a lower period.")
            raise IndexError

    def step(self, action):
        try:
            # Get reward for previous action
            reward = self.get_reward()

            # Get step timestamp
            timestamp = self.timestamp

            # Simulate portifolio rebalance
            self.simulate_trade(action, timestamp)

            # Calculate new portval
            self.portval = {'portval': self.calc_total_portval(), 'timestamp': self.portfolio_df.index[-1]}

            if self.index >= self.data_length - 2:
                done = True
                self.status["OOD"] += 1
            else:
                done = False

                # Get new index
            self.index += 1

            # Return new observation, reward, done flag and status for debugging
            return self.get_observation(True).astype('f'), reward, done, self.status

        except KeyboardInterrupt:
            self.status["OOD"] += 1
            # return self.get_observation(True).astype(np.float64), np.float64(0), False, self.status
            raise KeyboardInterrupt

        except Exception as e:
            self.logger.error(BacktestEnvironment.step, self.parse_error(e))
            if hasattr(self, 'email') and hasattr(self, 'psw'):
                self.send_email("TradingEnvironment Error: %s at %s" % (e,
                                datetime.strftime(self.timestamp, "%Y-%m-%d %H:%M:%S")),
                                self.parse_error(e))
            print("step action:", action)
            raise e


class PaperTradingEnvironment(TradingEnvironment):
    """
    Paper trading environment for financial strategies forward testing
    """
    def __init__(self, period, obs_steps, tapi, fiat, name):
        assert isinstance(tapi, PaperTradingDataFeed) or isinstance(tapi, Poloniex), "Paper trade tapi must be a instance of PaperTradingDataFeed."
        super().__init__(period, obs_steps, tapi, fiat, name)

    def reset(self):

        self.obs_df = pd.DataFrame()
        self.portfolio_df = pd.DataFrame()

        self.set_observation_space()
        self.set_action_space()

        self.balance = self.init_balance = self.get_balance()

        for symbol in self.symbols:
            self.tax[symbol] = convert_to.decimal(self.get_fee(symbol))

        obs = self.get_observation(True)

        self.action_df = pd.DataFrame([list(self.calc_portfolio_vector()) + [False]],
                                      columns=list(self.symbols) + ['online'],
                                      index=[self.timestamp])

        self.portval = {'portval': self.calc_total_portval(),
                        'timestamp': self.portfolio_df.index[-1]}

        return obs.astype(np.float64)

    def step(self, action):
        try:
            # Get reward for previous action
            reward = self.get_reward()

            # Get step timestamp
            timestamp = self.timestamp

            # Simulate portifolio rebalance
            self.simulate_trade(action, timestamp)

            done = True

            # Return new observation, reward, done flag and status for debugging
            return self.get_observation(True).astype('f'), reward, done, self.status

        except Exception as e:
            self.logger.error(PaperTradingEnvironment.step, self.parse_error(e))
            if hasattr(self, 'email'):
                self.send_email("TradingEnvironment Error: %s at %s" % (e,
                                datetime.strftime(self.timestamp, "%Y-%m-%d %H:%M:%S")),
                                self.parse_error(e))
            raise e


class LiveTradingEnvironment(TradingEnvironment):
    """
    Live trading environment for financial strategies execution
    """
    def __init__(self, period, obs_steps, tapi, fiat, name):
        assert isinstance(tapi, ExchangeConnection), "tapi must be an ExchangeConnection instance."
        super().__init__(period, obs_steps, tapi, fiat, name)

    # Data feed methods
    def get_balance(self):
        """
        Get last balance from exchange
        :return: dict: Dict containing Decimal values for portfolio allocation
        """
        try:
            balance = self.tapi.returnBalances()

            filtered_balance = {}
            for symbol in self.symbols:
                filtered_balance[symbol] = convert_to.decimal(balance[symbol])

            return filtered_balance

        except Exception as e:
            self.logger.error(TradingEnvironment.get_balance, self.parse_error(e))
            raise e

    def calc_total_portval(self, timestamp=None):
        """
        Calculate total portfolio value given last pair prices
        :param timestamp: For compatibility only
        :return: Decimal: Total portfolio value in fiat units
        """
        portval = convert_to.decimal('0E-8')
        balance = self.get_balance()
        ticker = self.tapi.returnTicker()
        for pair in self.pairs:
            portval = balance[pair.split('_')[1]].fma(convert_to.decimal(ticker[pair]['last']),
                                                      portval)
        portval += balance[self._fiat]

        return convert_to.decimal(portval)

    def calc_portfolio_vector(self):
        portfolio = np.empty(len(self.symbols), dtype=Decimal)
        portval = self.calc_total_portval()
        ticker = self.tapi.returnTicker()
        balance = self.get_balance()
        for i, pair in enumerate(self.pairs):
            portfolio[i] = safe_div(balance[pair.split('_')[1]] *
                                    convert_to.decimal(ticker[pair]['last']),  portval)

        portfolio[-1] = safe_div(balance[self._fiat], portval)

        return convert_to.decimal(portfolio)

    def get_balance_array(self):
        """
        Return ordered balance array
        :return: numpy ndarray:
        """
        balance_array = np.empty(len(self.symbols), dtype=Decimal)
        balance = self.get_balance()
        for i, symbol in enumerate(self.symbols):
            balance_array[i] = balance[symbol]
        return balance_array

    def get_desired_balance_array(self, action):
        """
        Return asset amounts given action array
        :param action: numpy ndarray: action array with norm summing one
        :return: numpy ndarray: asset amount array given action
        """
        desired_balance = np.empty(len(self.symbols), dtype=Decimal)
        portval = fiat = self.calc_total_portval()
        ticker = self.tapi.returnTicker()
        for i, pair in enumerate(self.pairs):
            desired_balance[i] = safe_div(portval * action[i],
                                          convert_to.decimal(ticker[pair]['last']))
            fiat -= portval * action[i]
        desired_balance[-1] = convert_to.decimal(fiat)

        return desired_balance

    def immediate_sell(self, symbol, amount):
        try:
            pair = self._fiat + '_' + symbol
            amount = str(amount)

            while True:
                try:
                    price = self.tapi.returnTicker()[pair]['highestBid']

                    self.logger.debug(LiveTradingEnvironment.immediate_sell,
                                      "Selling %s %s at %s" % (pair, amount, price))

                    response = self.tapi.sell(pair, price, amount, orderType="immediateOrCancel")

                    if 'amountUnfilled' in response:
                        self.logger.debug(LiveTradingEnvironment.immediate_sell,
                                          "Response: %s" % str(response))

                        if response['amountUnfilled'] == '0.00000000':
                            return True
                        else:
                            amount = response['amountUnfilled']

                except PoloniexError as error:
                    self.logger.debug(LiveTradingEnvironment.immediate_sell,
                                      self.parse_error(error))

                    if 'Total must be at least' in error.__str__():
                        return True

                    elif 'Amount must be at least' in error.__str__():
                        return True

                    elif 'Not enough %s.' % symbol == error.__str__():
                        amount = self.get_balance()[symbol]

                    else:
                        raise error

        except Exception as e:
            self.logger.error(LiveTradingEnvironment.immediate_sell, self.parse_error(e))
            if hasattr(self, 'email'):
                self.send_email("LiveTradingEnvironment Error: %s at %s" % (e,
                                                                            datetime.strftime(self.timestamp,
                                                                                              "%Y-%m-%d %H:%M:%S")),
                                self.parse_error(e))
            raise e

    def immediate_buy(self, symbol, amount):
        try:
            pair = self._fiat + '_' + symbol
            amount = str(amount)

            while True:
                try:
                    price = self.tapi.returnTicker()[pair]['lowestAsk']

                    self.logger.debug(LiveTradingEnvironment.immediate_buy,
                                      "Buying %s %s at %s" % (pair, amount, price))

                    response = self.tapi.buy(pair, price, amount, orderType="immediateOrCancel")

                    if 'amountUnfilled' in response:
                        self.logger.debug(LiveTradingEnvironment.immediate_buy,
                                          "Response: %s" % str(response))

                        if response['amountUnfilled'] == '0.00000000':
                            return True
                        else:
                            amount = response['amountUnfilled']

                except PoloniexError as error:
                    self.logger.debug(LiveTradingEnvironment.immediate_buy,
                                      self.parse_error(error))

                    if 'Total must be at least' in error.__str__():
                        return True

                    elif 'Amount must be at least' in error.__str__():
                        return True

                    elif 'Not enough %s.' % self._fiat == error.__str__():
                        if not self.status['NotEnoughFiat']:
                            self.status['NotEnoughFiat'] = True

                            price = convert_to.decimal(self.tapi.returnTicker()[pair]['lowestAsk'])
                            fiat_units = self.get_balance()[self._fiat]

                            amount = str(safe_div(fiat_units, price))

                        else:
                            raise error

                    else:
                        raise error

        except Exception as e:
            self.logger.error(LiveTradingEnvironment.immediate_buy, self.parse_error(e))
            if hasattr(self, 'email'):
                self.send_email("LiveTradingEnvironment Error: %s at %s" % (e,
                                                                            datetime.strftime(self.timestamp,
                                                                                              "%Y-%m-%d %H:%M:%S")),
                                self.parse_error(e))
            raise e

    # Online Trading methods
    def online_rebalance(self, action, timestamp):
        """
        Performs online portfolio rebalance within ExchangeConnection
        :param action: numpy array: action vector with desired portfolio weights. Norm must be one.
        :return: bool: True if fully executed; False otherwise.
        """
        try:
            self.status['NotEnoughFiat'] = False
            # First, assert action is valid
            action = self.assert_action(action)

            # Log desired action
            self.log_action_vector(timestamp, action, False)

            # Calculate position change given last porftolio and action vector
            balance_change = (self.get_desired_balance_array(action) - self.get_balance_array())[:-1]

            # Sell assets first
            for i, change in enumerate(balance_change):
                if change < 0:
                    symbol = self.symbols[i]
                    self.immediate_sell(symbol, abs(change))

            # Then, buy what you want
            for i, change in enumerate(balance_change):
                if change > 0:
                    symbol = self.symbols[i]
                    self.immediate_buy(symbol, abs(change))

            # Log executed action and final balance
            self.log_action_vector(self.timestamp, self.calc_portfolio_vector(), True)

            # Update portfolio_df
            final_balance = self.get_balance()
            final_balance['timestamp'] = timestamp
            self.balance = final_balance

            # Calculate new portval
            self.portval = {'portval': self.calc_total_portval(),
                            'timestamp': self.portfolio_df.index[-1]}

            return True

        except Exception as e:
            self.logger.error(LiveTradingEnvironment.online_rebalance, self.parse_error(e))
            if hasattr(self, 'email'):
                self.send_email("LiveTradingEnvironment Error: %s at %s" % (e,
                                datetime.strftime(self.timestamp, "%Y-%m-%d %H:%M:%S")),
                                self.parse_error(e))
            raise e

    # Env methods
    def reset(self):
        self.obs_df = pd.DataFrame()
        self.portfolio_df = pd.DataFrame()
        self.action_df = pd.DataFrame(columns=list(self.symbols) + ['online'], index=[self.timestamp])

        self.set_observation_space()
        self.set_action_space()

        self.balance = self.init_balance = self.get_balance()

        for symbol in self.symbols:
            self.tax[symbol] = convert_to.decimal(self.get_fee(symbol))

        obs = self.get_observation(True)

        self.portval = {'portval': self.calc_total_portval(),
                        'timestamp': self.portfolio_df.index[-1]}

        return obs.astype(np.float64)

    def step(self, action):
        try:
            # Get reward for previous action
            reward = self.get_reward()

            # Get step timestamp
            timestamp = self.timestamp

            # Simulate portifolio rebalance
            done = self.online_rebalance(action, timestamp)

            # Return new observation, reward, done flag and status for debugging
            return self.get_observation(True).astype(np.float64), np.float64(reward), done, self.status

        except Exception as e:
            self.logger.error(LiveTradingEnvironment.step, self.parse_error(e))
            if hasattr(self, 'email'):
                self.send_email("TradingEnvironment Error: %s at %s" % (e,
                                datetime.strftime(self.timestamp, "%Y-%m-%d %H:%M:%S")),
                                self.parse_error(e))
            raise e
