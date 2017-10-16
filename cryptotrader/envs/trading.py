"""
Trading environment class
data: 12/10/2017
author: Tau
"""
from .driver import *


class TradingEnvironment(Env):
    """
    Trading environment base class
    """
    ## Setup methods
    def __init__(self, tapi, name):
        assert isinstance(name, str), "Name must be a string"
        self.name = name

        if not os.path.exists('./logs'):
            os.makedirs('./logs')
        self.logger = Logger(self.name, './logs/')

        self.tapi = tapi

        self.epsilon = 1e-8
        self.status = None

        self._obs_steps = None
        self._freq = None

        # Portifolio info
        self._crypto = {}
        self._fiat = {}
        self.tax = {}
        self.pairs = []

        self.obs_df = None
        self.log_df = pd.DataFrame(index=[datetime.utcnow()])

        self.logger.info("Trading Environment initialization",
                         "Trading Environment Initialized!")

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

        self.log_df = self.log_df.append(pd.DataFrame(columns=self.symbols, index=[datetime.utcnow()]))

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
            return self._fiat['amount']
        except KeyError:
            self.logger.error(TradingEnvironment.fiat, "You must specify a fiat symbol first.")

    @fiat.setter
    def fiat(self, value):
        if isinstance(value, str):
            assert value in self.symbols, "Fiat symbol not in symbols."

            self._fiat['symbol'] = value
            self._fiat['amount'] = convert_to.decimal(self.get_balance()[value])
        elif isinstance(value, Decimal) or isinstance(value, float) or isinstance(value, int):
            self._fiat['amount'] = convert_to.decimal(value)

    @property
    def crypto(self):
        return self._crypto

    @crypto.setter
    def crypto(self, values):
        assert isinstance(values, dict), "Crypto value must be a dictionary containing the currencies balance."
        for symbol, value in values.items():
            if symbol not in self._fiat['symbol']:
                self._crypto[symbol] = convert_to.decimal(value)

    ## Data feed methods
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

    def get_ohlc(self, symbol):
        # TODO WRITE TEST
        ohlc_data = self.tapi.returnChartData(symbol, period=self.freq * 60,
                                              start=datetime.timestamp(datetime.utcnow() -
                                              timedelta(hours=3, minutes=self.freq * self.obs_steps + 1))
                                             ) #datetime.timestamp(datetime.utcnow()))

        ohlc_df = pd.DataFrame.from_records(ohlc_data)
        ohlc_df['date'] = ohlc_df.date.apply(
            lambda x: pd.to_datetime(datetime.strftime(datetime.fromtimestamp(x), "%Y-%m-%d %H:%M:%S")))
        ohlc_df.set_index('date', inplace=True)

        return ohlc_df[['open','high','low','close','volume']].asfreq("%dT" % self.freq).apply(convert_and_clean)

    def get_pair_history(self, pair):
        """
        Pools symbol's trade data from exchange
        """
        try:
            if self.freq < 5:
                df = self.get_ohlc_from_trades(pair)
            else:
                df = self.get_ohlc(pair)

            # If df is large enough, return
            if df.shape[0] >= self.obs_steps:
                return df.iloc[-self.obs_steps:]

            # Else, get more data and append
            else:
                raise ValueError("Dataframe is to small. Shape: %s" % str(df.shape))

        except Exception as e:
            self.logger.error(TradingEnvironment.get_pair_history, self.parse_error(e))
            return False

    def get_history(self):
        try:
            obs_list = []
            keys = []
            for symbol in self.pairs:
                keys.append(symbol)
                obs_list.append(self.get_pair_history(symbol))

            return pd.concat(obs_list, keys=keys, axis=1)

        except Exception as e:
            self.logger.error(TradingEnvironment.get_history, self.parse_error(e))
            return False

    def get_balance(self):
        try:
            balance = self.tapi.returnBalances()

            filtered_balance = {}
            for symbol in self.symbols:
                filtered_balance[symbol] = balance[symbol]

            return filtered_balance

        except Exception as e:
            self.logger.error(TradingEnvironment.get_history, self.parse_error(e))
            return False

    def update_observation(self):
        self.obs_df = self.get_history()

    # ## Trading methods
    def get_last_price(self, symbol):
        return self.obs_df.get_value(self.obs_df.index[-1], ("%s_%s" % (self._fiat['symbol'], symbol), 'close'))

    def _calc_total_portval(self):
        portval = convert_to.decimal('0.0')

        for symbol in self._crypto:
            portval += self.crypto[symbol] * self.get_last_price(symbol)
        portval += self.fiat

        return portval

    def store_value(self, timestamp, symbol, value):
        self.log_df.loc[timestamp, symbol] = convert_to.decimal(value)

    def _assert_action(self, action):
        # TODO WRITE TEST
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

    # Helper methods
    def _reset_status(self):
        self.status = {'OOD': False, 'Error': False, 'ValueError': False, 'ActionError': False}

    def parse_error(self, e):
        error_msg = '\n' + self.name + ' error -> ' + type(e).__name__ + ' in line ' + str(
            e.__traceback__.tb_lineno) + ': ' + str(e)
        return error_msg