"""
Trading environment class
data: 12/10/2017
author: Tau
"""
from .driver import *


class TradingEnvironment(Env):
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
        self.crypto = {}
        self.fiat = None
        self.tax = {}
        self.symbols = ['fiat']

        self.df = None

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
                if set(arg.split('_')).issubset(universe.keys()):
                    self.symbols.append(arg)
                else:
                    self.logger.error(TradingEnvironment.add_pairs, "Symbol not found on exchange currencies.")

            else:
                self.logger.error(TradingEnvironment.add_pairs, "Symbol name must be a string")

            if isinstance(arg, list):
                for item in arg:
                    if set(item.split('_')).issubset(universe.keys()):
                        if isinstance(item, str):
                            self.symbols.append(item)
                        else:
                            self.logger.error(TradingEnvironment.add_pairs, "Symbol name must be a string")

    # TODO
    def get_symbol_history(self, symbol):
        try:
            data = self.tapi.marketTradeHist(symbol)
            df = pd.DataFrame.from_records(data)

            while datetime.strptime(df.date[0],"%Y-%m-%d %H:%M:%S") + \
                    timedelta(minutes=self.freq * self.obs_steps) >= datetime.strptime(df.date[-1],"%Y-%m-%d %H:%M:%S"):


            # If df is large enough, return
            if df.shape[0] >= self.obs_steps:
                return df[::-1].iloc[-self.obs_steps:]

            # Else, get more data and append
            else:

                return df
        except Exception as e:
            self.logger.error(TradingEnvironment.get_symbol_history, self.parse_error(e))
            return False

    def get_history(self):
        pass

    def _reset_status(self):
        self.status = {'OOD': False, 'Error': False, 'ValueError': False, 'ActionError': False}

    def parse_error(self, e):
        error_msg = '\n' + self.name + ' error -> ' + type(e).__name__ + ' in line ' + str(
            e.__traceback__.tb_lineno) + ': ' + str(e)
        return error_msg