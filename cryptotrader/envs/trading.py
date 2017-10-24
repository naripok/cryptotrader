"""
Trading environment class
data: 12/10/2017
author: Tau
"""
from .driver import *
from decimal import DivisionByZero, InvalidOperation
from ..exchange_api.poloniex import PoloniexError
import json

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

        for pair in self.pairs:
            self.ohlc_data[pair] = pd.DataFrame.from_records(self.tapi.returnChartData(pair, period=self.period * 60,
                                                               start=start, end=end
                                                              ))
            # self.ohlc_data[pair]['date'] = self.ohlc_data[pair]['date'].apply(
            #     lambda x: datetime.fromtimestamp(int(x)))
            self.ohlc_data[pair].set_index('date', inplace=True, drop=False)

    def returnChartData(self, currencyPair, period, start=None, end=None):
        try:
            assert np.allclose(period, self.period * 60), "Invalid period"
            assert currencyPair in self.pairs, "Invalid pair"

            if not start:
                start = self.ohlc_data[currencyPair].date.index[-50]
            if not end:
                end = self.ohlc_data[currencyPair].date.index[-1]

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
    Data feeder for backtesting with TradingEnvironment.
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
        return self.tapi.returnFeeInfo()

    def returnCurrencies(self):
        return self.tapi.returnCurrencies()

    def returnChartData(self, currencyPair, period, start=None, end=None):
        return self.tapi.returnChartData(currencyPair, period, start=start, end=end)


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
        symbols = []
        for pair in self.pairs:
            symbol = pair.split('_')
            for s in symbol:
                symbols.append(s)

        return set(symbols)

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
                assert value in self.symbols, "Fiat not in symbols."
                self._fiat = value
                self._crypto = self.symbols.difference([self._fiat])

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
        return self.portfolio_df.ffill().iloc[-1, :].to_dict()

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
            if isinstance(value, Decimal) or isinstance(value, float) or isinstance(value, int):
                self.portfolio_df.at[self.timestamp, 'portval'] = convert_to.decimal(value)

            elif isinstance(value, dict):
                try:
                    timestamp = value['timestamp']
                except KeyError:
                    timestamp = self.timestamp
                self.portfolio_df.at[timestamp, 'portval'] = convert_to.decimal(value['portval'])

        except Exception as e:
            self.logger.error(TradingEnvironment.portval, self.parse_error(e))
            raise e

    ## Data feed methods
    @property
    def timestamp(self):
        #TODO FIX FOR DAYLIGHT SAVING TIME
        # return datetime.utcnow() - timedelta(hours=2)
        return datetime.fromtimestamp(time())

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

        self.portfolio_df = self.portfolio_df.append(pd.DataFrame(columns=self.symbols, index=[self.timestamp]))
        self.action_df = self.action_df.append(pd.DataFrame(columns=list(self.symbols)+['online'], index=[self.timestamp]))

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

    def get_ohlc_from_trades(self, pair, start=None, end=None):
        # TODO WRITE TEST
        df = self.get_pair_trades(pair, start=start, end=end)

        period = "%dmin" % self.period

        # Sample the trades into OHLC data
        df['rate'] = df['rate'].ffill().apply(convert_to.decimal)
        df['amount'] = df['amount'].apply(convert_to.decimal)
        df.index = df.date.apply(pd.to_datetime)

        # TODO REMOVE NANS
        index = df.resample(period).first().index
        out = pd.DataFrame(index=index)

        out['open'] = convert_and_clean(df['rate'].resample(period).first())
        out['high'] = convert_and_clean(df['rate'].resample(period).max())
        out['low'] = convert_and_clean(df['rate'].resample(period).min())
        out['close'] = convert_and_clean(df['rate'].resample(period).last())
        out['volume'] = convert_and_clean(df['amount'].resample(period).sum())

        return out

    def get_ohlc(self, symbol, start=None, end=None):
        # TODO WRITE TEST
        # TODO GET INVALID CANDLE TIMES RIGHT
        if start or end:
            ohlc_data = self.tapi.returnChartData(symbol, period=self.period * 60,
                                                  start=datetime.timestamp(start), end=datetime.timestamp(end)
                                                  )
        else:
            ohlc_data = self.tapi.returnChartData(symbol, period=self.period * 60,
                                                  start=datetime.timestamp(self.timestamp -
                                                                           timedelta(
                                                                               minutes=self.period * (self.obs_steps))),
                                                  end=datetime.timestamp(self.timestamp)
                                                  )

        ohlc_df = pd.DataFrame.from_records(ohlc_data)
        ohlc_df['date'] = ohlc_df.date.apply(
            lambda x: datetime.fromtimestamp(x))
        ohlc_df.set_index('date', inplace=True)

        return ohlc_df[['open','high','low','close',
                        'volume']].apply(convert_and_clean)

        # return ohlc_df[['open', 'high', 'low', 'close',
        #                 'quoteVolume']].asfreq("%dT" % self.freq).apply(convert_and_clean).rename(
        #     columns={'quoteVolume': 'volume'})

    def get_pair_history(self, pair, start=None, end=None):
        """
        Pools symbol's trade data from exchange api
        """
        # TODO RETURN POSITION ON OBS
        try:
            if self.period < 5:
                df = self.get_ohlc_from_trades(pair)
            else:
                df = self.get_ohlc(pair, start=start, end=end)

            if not start and not end:
                # If df is large enough, return
                while not df.shape[0] >= self.obs_steps:
                    sleep(5)
                    if self.period < 5:
                        df = self.get_ohlc_from_trades(pair)
                    else:
                        df = self.get_ohlc(pair, start=start, end=end)

                assert df.shape[0] >= self.obs_steps, "Dataframe is to small. Shape: %s" % str(df.shape)
                return df.iloc[-self.obs_steps:]

            else:
                return df

        except Exception as e:
            self.logger.error(TradingEnvironment.get_pair_history, self.parse_error(e))
            raise e

    def get_history(self, start=None, end=None, portifolio_vector=False):
        try:

            if not end:
                end = self.timestamp
            if not start:
                start = end - timedelta(minutes=self.period * (self.obs_steps))

            obs_list = []
            keys = []

            if portifolio_vector:
                port_vec = self.get_sampled_portfolio(start, end)

            for symbol in self.pairs:
                keys.append(symbol)
                history = self.get_pair_history(symbol, start=start, end=end)

                if portifolio_vector:
                    history = pd.concat([history,
                                         port_vec[symbol.split('_')[1]]],
                                         axis=1)
                obs_list.append(history)

            if portifolio_vector:
                keys.append(self._fiat)
                obs_list.append(port_vec[self._fiat])

            return pd.concat(obs_list, keys=keys, axis=1).ffill().bfill()

        except Exception as e:
            self.logger.error(TradingEnvironment.get_history, self.parse_error(e))
            raise e

    def get_observation(self, portfolio_vector=False):
        try:
            self.obs_df = self.get_history(portifolio_vector=portfolio_vector)
            return self.obs_df
        except Exception as e:
            self.logger.error(TradingEnvironment.get_observation, self.parse_error(e))
            raise e

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
        if isinstance(timestamp, pd.Timestamp):
            return self.obs_df.get_value(timestamp, ("%s_%s" % (self._fiat, symbol), 'close'))
        elif isinstance(timestamp, str) and timestamp == 'last' or timestamp is None:
            return self.obs_df.get_value(self.obs_df.index[-1], ("%s_%s" % (self._fiat, symbol), 'close'))

    def calc_total_portval(self, timestamp=None):
        portval = convert_to.decimal('0.0')

        for symbol in self._crypto:
            portval += self.get_crypto(symbol) * self.get_close_price(symbol, timestamp)
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
        for symbol in self.action_vector:
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
        for i, symbol in enumerate(self.action_vector):
            self.log_action(timestamp, symbol, vector[i])
        self.log_action(timestamp, 'online', online)

    def get_previous_portval(self):
        try:
            return self.portfolio_df.get_value(self.portfolio_df['portval'].last_valid_index(), 'portval')
        except Exception as e:
            self.logger.error(TradingEnvironment.get_previous_portval, self.parse_error(e))
            raise e

    def get_reward(self):
        # TODO TEST
        try:
            return (self.portval - self.get_previous_portval()) / self.get_previous_portval()

        except DivisionByZero:
            return (self.portval - self.get_previous_portval()) / (self.get_previous_portval() + self.epsilon)

        except InvalidOperation:
            return (self.portval - self.get_previous_portval()) / (self.get_previous_portval() + self.epsilon)

        except Exception as e:
            self.logger.error(TradingEnvironment.get_reward, self.parse_error(e))
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

    def set_action_vector(self):
        action_vector = []
        for pair in self.pairs:
            action_vector.append(pair.split('_')[1])
        action_vector.append(self._fiat)
        self.action_vector = action_vector

    def reset_status(self):
        self.status = {'OOD': False, 'Error': False, 'ValueError': False, 'ActionError': False}

    def reset(self):
        """
        Setup env with initial values
        :return:
        """
        self.set_observation_space()
        self.set_action_space()
        self.balance = self.get_balance()
        for symbol in self.symbols:
            self.tax[symbol] = convert_to.decimal(self.get_fee(symbol))
        obs = self.get_observation(True)
        self.set_action_vector()
        self.portval = self.calc_total_portval(self.obs_df.index[-1])
        return obs

    ## Analytics methods
    def get_sampled_portfolio(self, start, end):
        return self.portfolio_df.loc[start:end].resample("%dmin" % self.period).last()

    def get_sampled_actions(self, start, end):
        return self.action_df.loc[start:end].resample("%dmin" % self.period).last()

    def get_results(self, window=7):
        """
        Calculate arbiter desired actions statistics
        :return:
        """

        end = self.portfolio_df.index[-1]
        start = self.portfolio_df.index[0]

        self.results = self.get_sampled_portfolio(start, end).join(self.get_sampled_actions(start, end), rsuffix='_posit').ffill()

        obs = self.get_history(end=end, start=start)

        self.results['benchmark'] = convert_to.decimal('0e-8')
        self.results['returns'] = convert_to.decimal(np.nan)
        self.results['benchmark_returns'] = convert_to.decimal(np.nan)
        self.results['alpha'] = convert_to.decimal(np.nan)
        self.results['beta'] = convert_to.decimal(np.nan)
        self.results['drawdown'] = convert_to.decimal(np.nan)
        self.results['sharpe'] = convert_to.decimal(np.nan)

        ## Calculate benchmark portifolio, just equaly distribute money over all the assets
        # Calc init portval
        init_portval = Decimal('0E-8')
        init_time = self.results.index[1]
        for symbol in self._crypto:
            init_portval += self.get_sampled_portfolio(start, end).get_value(init_time, symbol) * \
                           obs.get_value(init_time, (self._fiat + '_' + symbol, 'close'))
        init_portval += self.get_sampled_portfolio(start, end).get_value(init_time, self._fiat)

        for symbol in self.pairs:
            self.results[symbol+'_benchmark'] = (Decimal('1') - self.tax[symbol.split('_')[1]]) * obs[symbol, 'close'] * \
                                        init_portval / (obs.at[init_time,
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
        self.results['drawdown'] = ec.roll_max_drawdown(self.results.returns, window=int(window))
        self.results['sharpe'] = ec.roll_sharpe_ratio(self.results.returns, window=int(window), risk_free=0.001)

        return self.results

    def plot_results(self):
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

        df = self.get_results().astype(np.float64)

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

        palettes = inferno(len(self.symbols))

        for i, symbol in enumerate(self.symbols):
            results[symbol + '_posit'] = p_pos.line(df.index, df[symbol + '_posit'], color=palettes[i], legend=symbol)

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

        results['handle'] = show(column(p_val, p_pos, p_ret, p_hist, p_sharpe, p_dd, p_alpha, p_beta),
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
            posit_change = (convert_to.decimal(action) - self.calc_portfolio_vector())[:-1]

            # Get initial portval
            portval = self.calc_total_portval()

            # Sell assets first
            for i, change in enumerate(posit_change):
                if change < convert_to.decimal('0E-8'):

                    symbol = self.action_vector[i]

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

                    symbol = self.action_vector[i]

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
                                datetime.strftime(datetime.fromtimestamp(time()), "%Y-%m-%d %H:%M:%S")),
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
            self.portval = {'portval': self.calc_total_portval(), 'timestamp': self.portfolio_df.index[-1]}

            done = True

            # Return new observation, reward, done flag and status for debugging
            return self.get_observation(True).astype(np.float64), np.float64(reward), done, self.status
        except Exception as e:
            self.logger.error(PaperTradingEnvironment.step, self.parse_error(e))
            if hasattr(self, 'email') and hasattr(self, 'psw'):
                self.send_email("TradingEnvironment Error: %s at %s" % (e,
                                datetime.strftime(datetime.fromtimestamp(time()), "%Y-%m-%d %H:%M:%S")),
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
        return datetime.fromtimestamp(self.tapi.ohlc_data[self.tapi.pairs[0]].index[self.index])

    def reset(self, reset_dfs=False):
        """
        Setup env with initial values
        :return:
        """
        # Reset index

        self.data_length = self.tapi.ohlc_data[list(self.tapi.ohlc_data.keys())[0]].shape[0]

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
        self.balance = self.get_balance()
        # Get fee values
        for symbol in self.symbols:
            self.tax[symbol] = convert_to.decimal(self.get_fee(symbol))
        obs = self.get_observation(True)
        # Get assets order
        self.set_action_vector()
        # Reset portfolio value
        self.portval = self.calc_total_portval(self.obs_df.index[-1])
        return obs

    def step(self, action):
        try:
            # Get new index
            self.index += 1

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

            # Return new observation, reward, done flag and status for debugging
            return self.get_observation(True).astype(np.float64), np.float64(reward), done, self.status
        except Exception as e:
            self.logger.error(TradingEnvironment.step, self.parse_error(e))
            if hasattr(self, 'email') and hasattr(self, 'psw'):
                self.send_email("TradingEnvironment Error: %s at %s" % (e,
                                datetime.strftime(datetime.fromtimestamp(time()), "%Y-%m-%d %H:%M:%S")),
                                self.parse_error(e))
            print(action)
            raise e
