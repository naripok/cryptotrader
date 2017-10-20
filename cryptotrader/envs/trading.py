"""
Trading environment class
data: 12/10/2017
author: Tau
"""
from .driver import *
from decimal import DivisionByZero, InvalidOperation

class TradingEnvironment(Env):
    """
    Trading environment base class
    """
    ## Setup methods
    def __init__(self, tapi, name):
        assert isinstance(name, str), "Name must be a string"
        self.name = name

        # Data feed api
        self.tapi = tapi

        # Environment configuration
        self.epsilon = convert_to.decimal('1E-8')
        self._obs_steps = None
        self._freq = None
        self.pairs = []
        self._crypto = []
        self._fiat = None
        self.tax = {}

        # Dataframes
        self.obs_df = pd.DataFrame(index=[self.timestamp])
        self.portfolio_df = pd.DataFrame(index=[self.timestamp])
        self.action_df = pd.DataFrame(index=[self.timestamp])

        # Logging and debugging
        self.status = None

        if not os.path.exists('./logs'):
            os.makedirs('./logs')
        self.logger = Logger(self.name, './logs/')
        self.logger.info("Trading Environment initialization",
                         "Trading Environment Initialized!")

    # Env properties
    @property
    def obs_steps(self):
        return self._obs_steps

    @obs_steps.setter
    def obs_steps(self, value):
        assert isinstance(value, int), "Obs steps must be a integer."
        assert value >= 3, "Obs steps must be >= 3. Value: %s" % str(value)
        self._obs_steps = value

    @property
    def freq(self):
        return self._freq

    @freq.setter
    def freq(self, value):
        assert isinstance(value, int) and value >= 1,\
            "Frequency must be a integer >= 1."
        self._freq = value

    def add_pairs(self, *args):

        universe = self.tapi.returnCurrencies()

        for arg in args:
            if isinstance(arg, str):
                if set(arg.split('_')).issubset(universe):
                    self.pairs.append(arg)
                else:
                    self.logger.error(TradingEnvironment.add_pairs, "Symbol not found on exchange currencies.")

            else:
                self.logger.error(TradingEnvironment.add_pairs, "Symbol name must be a string")

            if isinstance(arg, list):
                for item in arg:
                    if set(item.split('_')).issubset(universe):
                        if isinstance(item, str):
                            self.pairs.append(item)
                        else:
                            self.logger.error(TradingEnvironment.add_pairs, "Symbol name must be a string")

        self.portfolio_df = self.portfolio_df.append(pd.DataFrame(columns=self.symbols, index=[self.timestamp]))
        self.action_df = self.action_df.append(pd.DataFrame(columns=list(self.symbols)+['online'], index=[self.timestamp]))

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
                self.portfolio_df.loc[self.timestamp, self._fiat] = convert_to.decimal(value)

            elif isinstance(value, dict):
                try:
                    timestamp = value['timestamp']
                except KeyError:
                    timestamp = self.timestamp
                self.portfolio_df.loc[timestamp, self._fiat] = convert_to.decimal(value[self._fiat])

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
        return self.portfolio_df.iloc[-1].to_dict()

    @balance.setter
    def balance(self, values):
        try:
            assert isinstance(values, dict), "Balance must be a dictionary containing the currencies amount."
            timestamp = self.timestamp
            for symbol, value in values.items():
                self.portfolio_df.loc[timestamp, symbol] = convert_to.decimal(value)

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
                self.portfolio_df.loc[self.timestamp, 'portval'] = convert_to.decimal(value)

            elif isinstance(value, dict):
                try:
                    timestamp = value['timestamp']
                except KeyError:
                    timestamp = self.timestamp
                self.portfolio_df.loc[timestamp, 'portval'] = convert_to.decimal(value['portval'])

        except Exception as e:
            self.logger.error(TradingEnvironment.portval, self.parse_error(e))
            raise e

    ## Data feed methods
    @property
    def timestamp(self):
        #TODO FIX FOR DAYLIGHT SAVING TIME
        # return datetime.utcnow() - timedelta(hours=2)
        return datetime.fromtimestamp(time())

    def get_pair_trades(self, pair):
        # TODO WRITE TEST
        try:
            # Pool data from exchage
            data = self.tapi.marketTradeHist(pair)
            df = pd.DataFrame.from_records(data)

            # Get more data from exchange until have enough to make obs_steps rows
            while datetime.strptime(df.date.iloc[0], "%Y-%m-%d %H:%M:%S") - \
                    timedelta(minutes=self.freq * self.obs_steps) < \
                    datetime.strptime(df.date.iloc[-1], "%Y-%m-%d %H:%M:%S"):

                market_data = self.tapi.marketTradeHist(pair, end=datetime.timestamp(
                    datetime.strptime(df.date.iloc[-1], "%Y-%m-%d %H:%M:%S") - timedelta(hours=3)))

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
            return False

    def get_ohlc_from_trades(self, pair):
        # TODO WRITE TEST
        df = self.get_pair_trades(pair)

        freq = "%dmin" % self.freq

        # Sample the trades into OHLC data
        df['rate'] = df['rate'].ffill().apply(convert_to.decimal)
        df['amount'] = df['amount'].apply(convert_to.decimal)
        df.index = df.date.apply(pd.to_datetime)

        # TODO REMOVE NANS
        index = df.resample(freq).first().index
        out = pd.DataFrame(index=index)

        out['open'] = convert_and_clean(df['rate'].resample(freq).first())
        out['high'] = convert_and_clean(df['rate'].resample(freq).max())
        out['low'] = convert_and_clean(df['rate'].resample(freq).min())
        out['close'] = convert_and_clean(df['rate'].resample(freq).last())
        out['volume'] = convert_and_clean(df['amount'].resample(freq).sum())

        return out

    def get_ohlc(self, symbol, start=None, end=None):
        # TODO WRITE TEST
        # TODO GET INVALID CANDLE TIMES RIGHT
        if isinstance(start, float) or isinstance(end,float):
            ohlc_data = self.tapi.returnChartData(symbol, period=self.freq * 60,
                                                  start=start, end=end
                                                  )
        else:
            ohlc_data = self.tapi.returnChartData(symbol, period=self.freq * 60,
                                                  start=datetime.timestamp(self.timestamp -
                                                                           timedelta(
                                                                               minutes=self.freq * (self.obs_steps + 2)))
                                                  )

        ohlc_df = pd.DataFrame.from_records(ohlc_data)
        ohlc_df['date'] = ohlc_df.date.apply(
            lambda x: pd.to_datetime(datetime.strftime(datetime.fromtimestamp(x), "%Y-%m-%d %H:%M:%S")))
        ohlc_df.set_index('date', inplace=True)

        return ohlc_df[['open','high','low','close',
                        'quoteVolume']].asfreq("%dT" % self.freq).apply(convert_and_clean).rename(columns={'quoteVolume':'volume'})

    def get_pair_history(self, pair, start=None, end=None):
        """
        Pools symbol's trade data from exchange api
        """
        try:
            if self.freq < 5:
                df = self.get_ohlc_from_trades(pair)
            else:
                df = self.get_ohlc(pair, start=start, end=end)

            if not start or not end:
                # If df is large enough, return
                if df.shape[0] >= self.obs_steps:
                    return df.iloc[-self.obs_steps:]

                # Else, get more data and append
                else:
                    sleep(2)
                    if self.freq < 5:
                        df = self.get_ohlc_from_trades(pair)
                    else:
                        df = self.get_ohlc(pair, start=start, end=end)

                    if df.shape[0] >= self.obs_steps:
                        return df.iloc[-self.obs_steps:]
                    else:
                        raise ValueError("Dataframe is to small. Shape: %s" % str(df.shape))
            else:
                return df.iloc[-self.obs_steps:]

        except Exception as e:
            self.logger.error(TradingEnvironment.get_pair_history, self.parse_error(e))
            return False

    def get_history(self, start=None, end=None):
        try:
            obs_list = []
            keys = []
            for symbol in self.pairs:
                keys.append(symbol)
                obs_list.append(self.get_pair_history(symbol, start=start, end=end))

            return pd.concat(obs_list, keys=keys, axis=1).ffill()

        except ValueError:
            obs_list = []
            keys = []
            for symbol in self.pairs:
                keys.append(symbol)
                obs_list.append(self.get_pair_history(symbol, start=start, end=end))

            return pd.concat(obs_list, keys=keys, axis=1).ffill()

        except Exception as e:
            self.logger.error(TradingEnvironment.get_history, self.parse_error(e))
            return False

    def get_observation(self):
        try:
            self.obs_df = self.get_history()
            return self.obs_df
        except Exception as e:
            self.logger.error(TradingEnvironment.get_observation, self.parse_error(e))
            return False

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

    # ## Trading methods
    def get_close_price(self, symbol, timestamp=None):
        if isinstance(timestamp, pd.Timestamp):
            return self.obs_df.get_value(timestamp, ("%s_%s" % (self._fiat, symbol), 'close'))
        elif isinstance(timestamp, str) and timestamp == 'last' or timestamp is None:
            return self.obs_df.get_value(self.obs_df.index[-1], ("%s_%s" % (self._fiat, symbol), 'close'))

    def calc_total_portval(self, timestamp=None):
        portval = convert_to.decimal('0.0')

        for symbol in self._crypto:
            portval += self.crypto[symbol] * self.get_close_price(symbol, timestamp)
        portval += self.fiat

        return portval

    def calc_posit(self, symbol):
        if symbol not in self._fiat:
            return self.crypto[symbol] * self.get_close_price(symbol) / self.calc_total_portval()
        else:
            return self.fiat / self.calc_total_portval()

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
                    action /= action.sum()
                    try:
                        assert action.sum() == convert_to.decimal('1.0')
                    except AssertionError:
                        action[-1] += convert_to.decimal('1.0') - action.sum()
                        action /= action.sum()
                        assert action.sum() == convert_to.decimal('1.0')

                # if debug:
                #     self.logger.error(Apocalipse.assert_action, "Action does not belong to action space")

            assert action.sum() - convert_to.decimal('1.0') < convert_to.decimal('1e-6')

        except AssertionError:
            if debug:
                self.status['ActionError'] += 1
                # self.logger.error(Apocalipse.assert_action, "Action out of range")

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

    def log_action(self, timestamp, symbol, value):
        if symbol == 'online':
            self.action_df.loc[timestamp, symbol] = value
        else:
            self.action_df.loc[timestamp, symbol] = convert_to.decimal(value)

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
        try:
            return (self.portval - self.get_previous_portval()) / self.get_previous_portval()


        except DivisionByZero:
            return (self.portval - self.get_previous_portval()) / (self.get_previous_portval() + self.epsilon)

        except InvalidOperation:
            return (self.portval - self.get_previous_portval()) / (self.get_previous_portval() + self.epsilon)

        except Exception as e:
            self.logger.error(TradingEnvironment.get_reward, self.parse_error(e))
            raise e

    # Env methods
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
        self.logger.info(TrainingEnvironment.set_action_space, "Setting environment with %d symbols." % (len(self.symbols)))

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
        obs = self.get_observation()
        self.set_action_vector()
        self.portval = self.calc_total_portval(self.obs_df.index[-1])
        return obs

    # Analytics methods
    def get_sampled_portfolio(self):
        return self.portfolio_df.resample("%dmin" % self.freq).last()

    def _get_results(self, window=30):
        """
        Calculate arbiter desired actions statistics
        :return:
        """

        self.results = self.get_sampled_portfolio()

        obs = self.get_history(end=datetime.timestamp(self.results.index[-1]),
             start=datetime.timestamp(self.results.index[0]))

        self.results['benchmark'] = convert_to.decimal('0e-8')
        self.results['returns'] = convert_to.decimal(np.nan)
        self.results['benchmark_returns'] = convert_to.decimal(np.nan)
        self.results['alpha'] = convert_to.decimal(np.nan)
        self.results['beta'] = convert_to.decimal(np.nan)
        self.results['drawdown'] = convert_to.decimal(np.nan)
        self.results['sharpe'] = convert_to.decimal(np.nan)

        # Calculate benchmark portifolio, just equaly distribute money over all the assets

        for symbol in self.pairs:
            self.results[symbol+'_benchmark'] = (1 - self.tax[symbol.split('_')[1]]) * obs[symbol, 'close'] * \
                                        self.calc_total_portval(self.results.index[0]) / (obs.at[self.results.index[0],
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
    def __init__(self, tapi, name):
        super().__init__(tapi, name)

    def simulate_trade(self, action, timestamp):
        # Assert inputs
        action = self.assert_action(action)
        # for symbol in self._get_df_symbols(no_fiat=True): TODO FIX THIS
        #     self.observation_space.contains(observation[symbol])
        # assert isinstance(timestamp, pd.Timestamp)

        # Log action
        self.log_action_vector(timestamp, action, False)

        # Calculate position change given action
        posit_change = (convert_to.decimal(action) - self.calc_portfolio_vector())[:-1]

        # Get initial portval
        portval = self.calc_total_portval()

        # Sell assets first
        for i, change in enumerate(posit_change):
            if change < convert_to.decimal('0E-8'):

                symbol = self.action_vector[i]

                crypto_pool = portval * action[i] / self.get_close_price(symbol)

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

                crypto_pool = portval.fma(action[i], -fee) / self.get_close_price(symbol)

                self.crypto = {symbol: crypto_pool, 'timestamp': timestamp}

        # Log executed action
        self.log_action_vector(self.timestamp, self.calc_portfolio_vector(), True)

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

            done = False

            # Return new observation, reward, done flag and status for debugging
            return self.get_observation().astype(np.float64), np.float64(reward), done, self.status
        except Exception as e:
            self.logger.error(TradingEnvironment.step, self.parse_error(e))
            if hasattr(self, 'email') and hasattr(self, 'psw'):
                self.send_email("TradingEnvironment Error: %s at %s" % (e,
                                datetime.strftime(datetime.fromtimestamp(time()), "%Y-%m-%d %H:%M:%S")),
                                self.parse_error(e))
            return False