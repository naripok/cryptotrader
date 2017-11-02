"""
Trading environment class
data: 12/10/2017
author: Tau
"""
from ..core import Env
from ..spaces import *
from ..utils import Logger
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
from bokeh.models import HoverTool

from decimal import DivisionByZero, InvalidOperation
from ..exchange_api.poloniex import PoloniexError
import json

# Decimal precision
getcontext().prec = 24

# Debug flag
debug = True

class BacktestDataFeed(object):
    """
    Data feeder for backtesting with TradingEnvironment.
    """
    # TODO WRITE TESTS
    def __init__(self, tapi, period, pairs=[], portifolio={}):
        self.tapi = tapi
        self.ohlc_data = {}
        self.portfolio = portifolio
        self.pairs = pairs
        self.period = period
        self.data_length = 0

    @property
    def balance(self):
        return self.portfolio

    @balance.setter
    def balance(self, port):
        assert isinstance(port, dict), "Balance must be a dictionary with coin amounts."
        for key in port:
            self.portfolio[key] = port[key]

    def returnBalances(self):
        return self.portfolio

    def returnFeeInfo(self):
        return {'makerFee': '0.00150000',
                'nextTier': '600.00000000',
                'takerFee': '0.00250000',
                'thirtyDayVolume': '0.00000000'}

    def returnCurrencies(self):
        return self.tapi.returnCurrencies()

    def download_data(self, start=None, end=None):
        # TODO WRITE TEST
        self.ohlc_data = {}
        self.data_length = None
        for pair in self.pairs:
            self.ohlc_data[pair] = pd.DataFrame.from_records(self.tapi.returnChartData(pair, period=self.period * 60,
                                                               start=start, end=end
                                                              ))

        for key in self.ohlc_data:
            if not self.data_length or self.ohlc_data[key].shape[0] < self.data_length:
                self.data_length = self.ohlc_data[key].shape[0]

        for key in self.ohlc_data:
            if self.ohlc_data[key].shape[0] != self.data_length:
                self.ohlc_data[key] = pd.DataFrame.from_records(self.tapi.returnChartData(key, period=self.period * 60,
                                                               start=self.ohlc_data[key].date.iloc[-self.data_length],
                                                               end=end
                                                               ))


            # self.ohlc_data[pair]['date'] = self.ohlc_data[pair]['date'].apply(
            #     lambda x: datetime.fromtimestamp(int(x)))
            self.ohlc_data[key].set_index('date', inplace=True, drop=False)


        print("%d intervals, or %d days of data at %d minutes period downloaded." % (self.data_length, (self.data_length * self.period) /\
                                                                (24 * 60), self.period))

    def returnChartData(self, currencyPair, period, start=None, end=None):
        try:
            # assert np.allclose(period, self.period * 60), "Invalid period"
            assert currencyPair in self.pairs, "Invalid pair"
            #
            # if not start:
            #     start = self.ohlc_data[currencyPair].date.index[-50]
            # if not end:
            #     end = self.ohlc_data[currencyPair].date.index[-1]

            # Faster method
            # data = []
            # for row in self.ohlc_data[currencyPair].loc[start:end, :].iterrows():
            #     data.append(row[1].to_dict())

            data = json.loads(self.ohlc_data[currencyPair].loc[start:end, :].to_json(orient='records'))

            return data

        except AssertionError as e:
            if "Invalid period" == e:
                raise PoloniexError("%d invalid candle period" % period)
            elif "Invalid pair" == e:
                raise PoloniexError("Invalid currency pair.")


class PaperTradingDataFeed(object):
    """
    Data feeder for paper trading with TradingEnvironment.
    """
    # TODO WRITE TESTS
    def __init__(self, tapi, period, pairs=[], portifolio={}):
        self.tapi = tapi
        self.portfolio = portifolio
        self.pairs = pairs
        self.period = period

    @property
    def balance(self):
        return self.portfolio

    @balance.setter
    def balance(self, port):
        assert isinstance(port, dict), "Balance must be a dictionary with coin amounts."
        for key in port:
            self.portfolio[key] = port[key]

    def returnBalances(self):
        return self.portfolio

    def returnFeeInfo(self):
        # return self.tapi.returnFeeInfo()
        return {'makerFee': '0.00150000',
                'nextTier': '600.00000000',
                'takerFee': '0.00250000',
                'thirtyDayVolume': '0.00000000'}

    def returnCurrencies(self):
        return self.tapi.returnCurrencies()

    def returnChartData(self, currencyPair, period, start=None, end=None):
        try:
            return self.tapi.returnChartData(currencyPair, period, start=start, end=end)

        except PoloniexError("Invalid json response returned"):
            raise ValueError("Bad exchange response data.")


class TradingEnvironment(Env):
    """
    Trading environment base class
    """
    ## Setup methods
    def __init__(self, period, obs_steps, tapi, name="TradingEnvironment"):
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
        self.status = {'OOD': False, 'Error': False, 'ValueError': False, 'ActionError': False}

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
        if not self._symbols:
            symbols = []
            for pair in self.pairs:
                symbols.append(pair.split('_')[1])
            symbols.append(self._fiat)
            self._symbols = symbols
            return self._symbols
        else:
            return self._symbols

    @property
    def fiat(self):
        try:
            return self.portfolio_df.get_value(self.portfolio_df[self._fiat].last_valid_index(), self._fiat)
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
                for symbol in self.pairs:
                    symbols += symbol.split('_')
                symbols = set(symbols)
                assert value in symbols, "Fiat not in symbols."
                self._fiat = value
                self._crypto = symbols.difference([self._fiat])

            elif isinstance(value, Decimal) or isinstance(value, float) or isinstance(value, int):
                self.portfolio_df.at[self.timestamp, self._fiat] = convert_to.decimal(value)

            elif isinstance(value, dict):
                try:
                    timestamp = value['timestamp']
                except KeyError:
                    timestamp = self.timestamp
                self.portfolio_df.at[timestamp, self._fiat] = convert_to.decimal(value[self._fiat])

        except Exception as e:
            self.logger.error(TradingEnvironment.fiat, self.parse_error(e))
            raise e

    @property
    def crypto(self):
        try:
            crypto = {}
            for symbol in self._crypto:
                crypto[symbol] = self.portfolio_df.get_value(self.portfolio_df[symbol].last_valid_index(), symbol)
            return crypto
        except KeyError as e:
            self.logger.error(TradingEnvironment.crypto, "No valid value on portfolio dataframe.")
            raise e
        except Exception as e:
            self.logger.error(TradingEnvironment.crypto, self.parse_error(e))
            raise e

    def get_crypto(self, symbol):
        try:
            return self.portfolio_df.get_value(self.portfolio_df[symbol].last_valid_index(), symbol)
        except KeyError as e:
            self.logger.error(TradingEnvironment.crypto, "No valid value on portfolio dataframe.")
            raise e
        except Exception as e:
            self.logger.error(TradingEnvironment.crypto, self.parse_error(e))
            raise e

    @crypto.setter
    def crypto(self, values):
        try:
            assert isinstance(values, dict), "Crypto value must be a dictionary containing the currencies balance."
            try:
                timestamp = values['timestamp']
            except KeyError:
                timestamp = self.timestamp
            for symbol, value in values.items():
                if symbol not in [self._fiat, 'timestamp']:
                    self.portfolio_df.at[timestamp, symbol] = convert_to.decimal(value)

        except Exception as e:
            self.logger.error(TradingEnvironment.crypto, self.parse_error(e))
            raise e

    @property
    def balance(self):
        return self.portfolio_df.ffill().loc[self.portfolio_df.index[-1], self.symbols].to_dict()

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
            # if isinstance(value, Decimal) or isinstance(value, float) or isinstance(value, int):
            #     self.portfolio_df.at[self.timestamp, 'portval'] = convert_to.decimal(value)
            #
            # elif isinstance(value, dict):
            #     try:
            #         timestamp = value['timestamp']
            #     except KeyError:
            #         timestamp = self.timestamp
            #     self.portfolio_df.at[timestamp, 'portval'] = convert_to.decimal(value['portval'])

            self.portfolio_df.at[value['timestamp'], 'portval'] = convert_to.decimal(value['portval'])
        except KeyError:
            self.portfolio_df.at[self.timestamp, 'portval'] = convert_to.decimal(value['portval'])
        except TypeError:
            self.portfolio_df.at[self.timestamp, 'portval'] = convert_to.decimal(value)

        except Exception as e:
            self.logger.error(TradingEnvironment.portval, self.parse_error(e))
            raise e

    ## Data feed methods
    @property
    def timestamp(self):
        #TODO FIX FOR DAYLIGHT SAVING TIME
        return datetime.now(timezone.utc)

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

        # self.portfolio_df = self.portfolio_df.append(pd.DataFrame(columns=self.symbols, index=[self.timestamp]))
        # self.action_df = self.action_df.append(pd.DataFrame(columns=list(self.symbols)+['online'], index=[self.timestamp]))

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

    def get_ohlc(self, symbol, index):
        # TODO WRITE TEST
        # TODO GET INVALID CANDLE TIMES RIGHT

        start = index[0]
        end = index[-1]

        ohlc_df = pd.DataFrame.from_records(self.tapi.returnChartData(symbol,
                                                                        period=self.period * 60,
                                                                        start=datetime.timestamp(start),
                                                                        end=datetime.timestamp(end)))
        # TODO 1 FIND A BETTER WAY
        ohlc_df.set_index(ohlc_df.date.apply(lambda x: datetime.fromtimestamp(x).astimezone(timezone.utc)), inplace=True)



        return ohlc_df[['open','high','low','close',
                        'volume']].reindex(index).asfreq("%dT" % self.period)

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

    def get_history(self, start=None, end=None, portfolio_vector=False):
        try:
            obs_list = []
            keys = []
            is_bounded = True
            if not end:
                end = self.timestamp
                is_bounded = False
            if not start:
                start = end - timedelta(minutes=self.period * self.obs_steps)
                index = pd.date_range(start=start.astimezone(timezone.utc),
                                      end=end.astimezone(timezone.utc),
                                      freq="%dT" % self.period).ceil("%dT" % self.period)[-self.obs_steps:]
                is_bounded = False
            else:
                index = pd.date_range(start=start.astimezone(timezone.utc),
                                      end=end.astimezone(timezone.utc),
                                      freq="%dT" % self.period).ceil("%dT" % self.period)

            if portfolio_vector:
                port_vec = self.get_sampled_portfolio(index)
                for symbol in self.pairs:
                    keys.append(symbol)
                    history = self.get_ohlc(symbol, index)
                    history = pd.concat([history, port_vec[symbol.split('_')[1]]], axis=1)
                    obs_list.append(history)
                keys.append(self._fiat)
                obs_list.append(port_vec[self._fiat])

                obs = pd.concat(obs_list, keys=keys, axis=1)

                cols_to_bfill = [col for col in zip(self.pairs, self.symbols)] + [(self._fiat, self._fiat)]
                obs = obs.fillna(obs[cols_to_bfill].bfill())

                if not is_bounded:
                    assert obs.shape[0] >= self.obs_steps, "Dataframe is to small. Shape: %s" % str(obs.shape)

                return obs.ffill().apply(convert_to.decimal, raw=True)
            else:
                for symbol in self.pairs:
                    keys.append(symbol)
                    history = self.get_ohlc(symbol, index)
                    obs_list.append(history)

                obs = pd.concat(obs_list, keys=keys, axis=1)

                if not is_bounded:
                    assert obs.shape[0] >= self.obs_steps, "Dataframe is to small. Shape: %s" % str(obs.shape)

                return obs.ffill().apply(convert_to.decimal, raw=True)

        except Exception as e:
            self.logger.error(TradingEnvironment.get_history, self.parse_error(e))
            raise e

    def get_observation(self, portfolio_vector=False):
        try:
            self.obs_df = self.get_history(portfolio_vector=portfolio_vector)
            return self.obs_df

        except PoloniexError:
            sleep(int(self.period * 30))
            self.obs_df = self.get_history(portfolio_vector=portfolio_vector)
            return self.obs_df

        except Exception as e:
            self.logger.error(TradingEnvironment.get_observation, self.parse_error(e))
            raise e

    def get_sampled_actions(self, index=None):
        if index is None:
            start = self.portfolio_df.index[0]
            end = self.portfolio_df.index[-1]

        else:
            start = index[0]
            end = index[-1]

        # TODO 1 FIND A BETTER WAY
        return self.action_df.loc[start:end].resample("%dmin" % self.period).last()

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

    ## Trading methods
    def get_close_price(self, symbol, timestamp=None):
        if not timestamp:
            timestamp = self.obs_df.index[-1]
        return self.obs_df.get_value(timestamp, ("%s_%s" % (self._fiat, symbol), 'close'))

    def calc_total_portval(self, timestamp=None):
        portval = convert_to.decimal('0E-8')

        for symbol in self._crypto:
            portval = self.get_crypto(symbol).fma(self.get_close_price(symbol, timestamp), portval)
        portval += self.fiat

        return portval

    def calc_posit(self, symbol):
        if symbol not in self._fiat:
            try:
                return self.get_crypto(symbol) * self.get_close_price(symbol) / self.calc_total_portval()
            except DivisionByZero:
                return self.get_crypto(symbol) * self.get_close_price(symbol) / (self.calc_total_portval() + self.epsilon)
            except InvalidOperation:
                return self.get_crypto(symbol) * self.get_close_price(symbol) / (self.calc_total_portval() + self.epsilon)
        else:
            try:
                return self.fiat / self.calc_total_portval()
            except DivisionByZero:
                return self.fiat / (self.calc_total_portval() + self.epsilon)
            except InvalidOperation:
                return self.fiat / (self.calc_total_portval() + self.epsilon)

    def calc_portfolio_vector(self):
        portfolio = []
        for symbol in self.symbols:
            portfolio.append(self.calc_posit(symbol))
        return np.array(portfolio)

    def assert_action(self, action):
        # TODO WRITE TEST
        try:
            for posit in action:
                if not isinstance(posit, Decimal):
                    if isinstance(posit, np.float32):
                        action = convert_to.decimal(np.float64(action))
                    else:
                        action = convert_to.decimal(action)

            try:
                assert self.action_space.contains(action)

            except AssertionError:
                # normalize
                if action.sum() != convert_to.decimal('1.0'):
                    try:
                        action /= action.sum()
                    except DivisionByZero:
                        action /= (action.sum() + self.epsilon)
                    except InvalidOperation:
                        action /= (action.sum() + self.epsilon)

                    try:
                        assert action.sum() == convert_to.decimal('1.0')
                    except AssertionError:
                        action[-1] += convert_to.decimal('1.0') - action.sum()
                        try:
                            action /= action.sum()
                        except DivisionByZero:
                            action /= (action.sum() + self.epsilon)
                        except InvalidOperation:
                            action /= (action.sum() + self.epsilon)

                        assert action.sum() == convert_to.decimal('1.0')

                # if debug:
                #     self.logger.error(Apocalipse.assert_action, "Action does not belong to action space")

            assert action.sum() - convert_to.decimal('1.0') < convert_to.decimal('1e-6')

        except AssertionError:
            if debug:
                self.status['ActionError'] += 1
                # self.logger.error(Apocalipse.assert_action, "Action out of range")
            try:
                action /= action.sum()
            except DivisionByZero:
                action /= (action.sum() + self.epsilon)
            except InvalidOperation:
                action /= (action.sum() + self.epsilon)

            try:
                assert action.sum() == convert_to.decimal('1.0')
            except AssertionError:
                action[-1] += convert_to.decimal('1.0') - action.sum()
                try:
                    assert action.sum() == convert_to.decimal('1.0')
                except AssertionError:
                    try:
                        action /= action.sum()
                    except DivisionByZero:
                        action /= (action.sum() + self.epsilon)
                    except InvalidOperation:
                        action /= (action.sum() + self.epsilon)

        except Exception as e:
            self.logger.error(TradingEnvironment.assert_action, self.parse_error(e))
            raise e

        return action

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
            return self.portfolio_df.get_value(self.portfolio_df['portval'].last_valid_index(), 'portval')
        except Exception as e:
            self.logger.error(TradingEnvironment.get_last_portval, self.parse_error(e))
            raise e

    def get_reward(self):
        # TODO TEST
        # try:
        return self.portval / (self.get_last_portval() + self.epsilon)

        # except DivisionByZero:
        #     return self.portval / (self.get_last_portval() + self.epsilon)
        #
        # except InvalidOperation:
        #     return self.portval / (self.get_last_portval() + self.epsilon)

        # except Exception as e:
        #     self.logger.error(TradingEnvironment.get_reward, self.parse_error(e))
        #     raise e

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
        self.status = {'OOD': False, 'Error': False, 'ValueError': False, 'ActionError': False}

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
        self.results = self.get_sampled_portfolio().join(self.get_sampled_actions(), rsuffix='_posit').ffill()

        obs = self.get_history(self.results.index[0], self.results.index[-1])

        self.results['benchmark'] = convert_to.decimal('1E-8')
        self.results['returns'] = convert_to.decimal(np.nan)
        self.results['benchmark_returns'] = convert_to.decimal(np.nan)
        self.results['alpha'] = convert_to.decimal(np.nan)
        self.results['beta'] = convert_to.decimal(np.nan)
        self.results['drawdown'] = convert_to.decimal(np.nan)
        self.results['sharpe'] = convert_to.decimal(np.nan)

        ## Calculate benchmark portifolio, just equaly distribute money over all the assets
        # Calc init portval
        init_portval = Decimal('1E-8')
        init_time = self.results.index[0]
        for symbol in self._crypto:
            init_portval += convert_to.decimal(self.init_balance[symbol]) * \
                           obs.get_value(init_time, (self._fiat + '_' + symbol, 'close'))
        init_portval += convert_to.decimal(self.init_balance[self._fiat])

        with localcontext() as ctx:
            ctx.rounding = ROUND_UP
            for symbol in self.pairs:
                self.results[symbol+'_benchmark'] = (Decimal('1') - self.tax[symbol.split('_')[1]]) * obs[symbol, 'close'] * \
                                            init_portval / (obs.get_value(init_time,
                                            (symbol, 'close')) * Decimal(self.action_space.low.shape[0] - 1))
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
                       plot_width=800, plot_height=300,
                       tools=['crosshair','reset','xwheel_zoom','pan,box_zoom', pos_hover],
                       toolbar_location="above"
                       )
        config_fig(p_pos)

        palettes = inferno(len(self.symbols))

        for i, symbol in enumerate(self.symbols):
            results[symbol + '_posit'] = p_pos.line(df.index, df[symbol + '_posit'], color=palettes[i],
                                                    legend=symbol, line_width=1.2)#, muted_color=palettes[i], muted_alpha=0.2)
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
                       plot_width=800, plot_height=400,
                       tools=['crosshair', 'reset', 'xwheel_zoom', 'pan,box_zoom', val_hover],
                       toolbar_location="above"
                       )
        config_fig(p_val)

        results['portval'] = p_val.line(df.index, df.portval, color='green', line_width=1.2, legend='portval')
        results['benchmark'] = p_val.line(df.index, df.benchmark, color='red', line_width=1.2, legend="benchmark")
        p_val.legend.click_policy = "hide"

        # Individual assets portval
        p_pval = figure(title="Pair Performance",
                       x_axis_type="datetime",
                       x_axis_label='timestep',
                       y_axis_label='position',
                       plot_width=800, plot_height=400,
                       tools=['crosshair', 'reset', 'xwheel_zoom', 'pan,box_zoom', val_hover],
                       toolbar_location="above"
                       )
        config_fig(p_pval)

        for i, symbol in enumerate(self.pairs):
            results[symbol+'_benchmark'] = p_pval.line(df.index, df[symbol+'_benchmark'], color=palettes[i], line_width=1.2,
                                                      legend=symbol)
            p_pval.legend.click_policy = "hide"

        # Portifolio and benchmark returns
        p_ret = figure(title="Portifolio / Benchmark Returns",
                       x_axis_type="datetime",
                       x_axis_label='timestep',
                       y_axis_label='Returns',
                       plot_width=800, plot_height=200,
                       tools=['crosshair','reset','xwheel_zoom','pan,box_zoom'],
                       toolbar_location="above"
                       )
        config_fig(p_ret)

        results['bench_ret'] = p_ret.line(df.index, df.benchmark_returns, color='red', line_width=1.2)
        results['port_ret'] = p_ret.line(df.index, df.returns, color='green', line_width=1.2)

        p_hist = figure(title="Portifolio Value Pct Change Distribution",
                        x_axis_label='Pct Change',
                        y_axis_label='frequency',
                        plot_width=800, plot_height=300,
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
                         plot_width=800, plot_height=200,
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
                        plot_width=800, plot_height=200,
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
                      plot_width=800, plot_height=200,
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
                          plot_width=800, plot_height=200,
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

    def send_email(self, subject, body):
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
            self.logger.error(TradingEnvironment.send_email, self.parse_error(e))
        finally:
            pass


class PaperTradingEnvironment(TradingEnvironment):
    """
    Paper trading environment for financial strategies forward testing
    """
    def __init__(self, period, obs_steps, tapi, name):
        super().__init__(period, obs_steps, tapi, name)

    def reset(self):
        self.obs_df = pd.DataFrame()
        self.portfolio_df = pd.DataFrame()
        self.set_observation_space()
        self.set_action_space()
        self.balance = self.init_balance = self.get_balance()
        for symbol in self.symbols:
            self.tax[symbol] = convert_to.decimal(self.get_fee(symbol))
        obs = self.get_observation(True)
        self.action_df = self.action_df.append(
            pd.DataFrame(columns=list(self.symbols) + ['online'], index=[self.timestamp]))
        self.portval = self.calc_total_portval(self.obs_df.index[-1])
        return obs.astype(np.float64)

    def simulate_trade(self, action, timestamp):
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

                    try:
                        crypto_pool = portval * action[i] / self.get_close_price(symbol)
                    except DivisionByZero:
                        crypto_pool = portval * action[i] / (self.get_close_price(symbol) + self.epsilon)
                    except InvalidOperation:
                        crypto_pool = portval * action[i] / (self.get_close_price(symbol) + self.epsilon)

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
                    try:
                        assert self.fiat >= convert_to.decimal('0E-8')
                    except AssertionError:
                        portval += self.fiat
                        self.fiat = {self._fiat: convert_to.decimal('0E-8'), 'timestamp': timestamp}

                    with localcontext() as ctx:
                        ctx.rounding = ROUND_UP

                        fee = self.tax[symbol] * portval * change

                    try:
                        crypto_pool = portval.fma(action[i], -fee) / self.get_close_price(symbol)
                    except DivisionByZero:
                        crypto_pool = portval.fma(action[i], -fee) / (self.get_close_price(symbol) + self.epsilon)
                    except InvalidOperation:
                        crypto_pool = portval.fma(action[i], -fee) / (self.get_close_price(symbol) + self.epsilon)

                    self.crypto = {symbol: crypto_pool, 'timestamp': timestamp}

            # Log executed action and final balance
            self.log_action_vector(self.timestamp, self.calc_portfolio_vector(), True)
            final_balance = self.balance
            final_balance['timestamp'] = timestamp
            self.balance = final_balance

        except Exception as e:
            self.logger.error(PaperTradingEnvironment.simulate_trade, self.parse_error(e))
            if hasattr(self, 'email') and hasattr(self, 'psw'):
                self.send_email("TradingEnvironment Error: %s at %s" % (e,
                                datetime.strftime(self.timestamp, "%Y-%m-%d %H:%M:%S")),
                                self.parse_error(e))
            raise e

    def step(self, action):
        try:
            # Get reward for previous action
            reward = self.get_reward()

            # Get step timestamp
            timestamp = self.timestamp

            # Simulate portifolio rebalance
            self.simulate_trade(action, timestamp)

            # Calculate new portval
            self.portval = {'portval': self.calc_total_portval(),
                            'timestamp': self.portfolio_df.index[-1]}

            done = True

            # Return new observation, reward, done flag and status for debugging
            return self.get_observation(True).astype(np.float64), np.float64(reward), done, self.status
        except Exception as e:
            self.logger.error(PaperTradingEnvironment.step, self.parse_error(e))
            if hasattr(self, 'email') and hasattr(self, 'psw'):
                self.send_email("TradingEnvironment Error: %s at %s" % (e,
                                datetime.strftime(self.timestamp, "%Y-%m-%d %H:%M:%S")),
                                self.parse_error(e))
            raise e


class BacktestEnvironment(PaperTradingEnvironment):
    """
    Backtest environment for financial strategies history testing
    """
    def __init__(self, period, obs_steps, tapi, name):
        assert isinstance(tapi, BacktestDataFeed), "Backtest tapi must be a instance of BacktestDataFeed."
        self.index = obs_steps
        super().__init__(period, obs_steps, tapi, name)
        self.data_length = None
        self.training = False

    @property
    def timestamp(self):
        return datetime.fromtimestamp(self.tapi.ohlc_data[self.tapi.pairs[0]].index[self.index]).astimezone(timezone.utc)

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
                self.index = np.random.random_integers(self.obs_steps, self.data_length - 2)
            else:
                self.index = self.obs_steps

            # Reset log dfs
            if reset_dfs:
                self.obs_df = pd.DataFrame()
                self.portfolio_df = pd.DataFrame()
                self.action_df = pd.DataFrame(columns=list(self.symbols)+['online'], index=[self.timestamp])

            # Set spaces
            self.set_observation_space()
            self.set_action_space()

            # Reset balance
            self.balance = self.init_balance = self.get_balance()

            # Get fee values
            for symbol in self.symbols:
                self.tax[symbol] = convert_to.decimal(self.get_fee(symbol))
            obs = self.get_observation(True)

            # Reset portfolio value
            self.portval = self.calc_total_portval(self.obs_df.index[-1])

            # Return first observation
            return obs.astype(np.float64)

        except IndexError:
            print("Insufficient tapi data. You must choose a bigger time span.")
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
            return self.get_observation(True).astype(np.float64), np.float64(reward), done, self.status

        except KeyboardInterrupt:
            self.status["OOD"] += 1
            # return self.get_observation(True).astype(np.float64), np.float64(0), False, self.status
            raise KeyboardInterrupt

        except Exception as e:
            self.logger.error(TradingEnvironment.step, self.parse_error(e))
            if hasattr(self, 'email') and hasattr(self, 'psw'):
                self.send_email("TradingEnvironment Error: %s at %s" % (e,
                                datetime.strftime(self.timestamp, "%Y-%m-%d %H:%M:%S")),
                                self.parse_error(e))
            print(action)
            raise e
