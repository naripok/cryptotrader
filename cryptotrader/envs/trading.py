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

        # Portifolio info
        self.crypto = {}
        self.fiat = None
        self.tax = {}
        self.symbols = ['fiat']

        self.df = None

        self.logger.info("Trading Environment initialization",
                         "Trading Environment Initialized!")

    def filter_universe(self, **args):

        universe = self.tapi.returnCurrencies()

        for arg in args:
            if isinstance(arg, str):
                if arg in universe.keys():
                    self.symbols.append(arg)
                else:
                    self.logger.error(TradingEnvironment.filter_universe, "Symbol not found on exchange currencies.")

            else:
                self.logger.error(TradingEnvironment.filter_universe, "Symbol name must be a string")



            if isinstance(arg, list):
                for item in arg:
                    if item in universe.keys():
                        if isinstance(item, str):
                            self.symbols.append(item)
                        else:
                            self.logger.error(TradingEnvironment.filter_universe, "Symbol name must be a string")

    def get_history(self):
        return NotImplementedError

    def _reset_status(self):
        self.status = {'OOD': False, 'Error': False, 'ValueError': False, 'ActionError': False}

