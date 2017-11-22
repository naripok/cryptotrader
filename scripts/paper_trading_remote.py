"""
Paper trading script
author: T
data: 21/11/2017
"""

# imports
import sys
sys.path.insert(0, '../')
import logging
logging.basicConfig(level=logging.ERROR)
from cryptotrader.envs.trading import PaperTradingEnvironment, PaperTradingDataFeed
from cryptotrader.agents import apriori
from cryptotrader.exchange_api.poloniex import Poloniex

if __name__ == '__main__':
    ## Simulation Params
    test_name = 'name'
    obs_steps = 3 # Observation steps
    period = 120 # Observation period

    ## Universe declaration
    # Universe
    pairs = [
                "USDT_BTC",
                "USDT_ETH",
                "USDT_LTC",
                "USDT_XRP",
                "USDT_XMR",
                "USDT_ETC",
                "USDT_ZEC",
                "USDT_DASH",
                "USDT_BCH"
            ]

    # Quote symbol
    fiat_symbol = 'USDT'

    # pairs = [
    #           "BTC_USDT",
    #             "BTC_ETH",
    #             "BTC_LTC",
    #             "BTC_XRP",
    #             "BTC_XMR",
    #             "BTC_ETC",
    #             "BTC_ZEC",
    #             "BTC_DASH"
    #         ]
    # fiat_symbol = "BTC"

    # Balance setup
    # init_funds = make_balance(crypto=0.0, fiat=100.0, pairs=pairs) # Initial portfolio

    init_funds = {
                    "BTC": "0.01170000",
                    "BCH": "0.07960000",
                    "ETH": "0.02400000",
                    "ETC": "3.99710000",
                    "LTC": "1.40740000",
                    "XRP": "135.5694000",
                    "XMR": "0.00000000",
                    "ZEC": "0.10990000",
                    "DASH": "0.00000000",
                    "USDT": "0.00000000"
                }

    # Email data for reporting
    emails = {'email@gmail.com': 'password', 'email2@gmail.com': 'password'}

    ## Env setup
    # Setup exchange connection
    papi = Poloniex()
    tapi = PaperTradingDataFeed(papi, period, pairs, init_funds)

    # Instantiate and setup simulation environment
    env = PaperTradingEnvironment(period, obs_steps, tapi, test_name)
    env.add_pairs(pairs)
    env.fiat = fiat_symbol

    # Email for report generation (Gmail only)
    env.set_email(emails)

    # Reset simulation and get first observation
    obs = env.reset()

    # Instantiate and setup agent
    # agent = apriori.TestAgent(env.get_observation(True).shape)
    agent = apriori.PAMRTrader(sensitivity=0.02, alpha=3, C=None, variant='PAMR', name=test_name)

    # Run simulation
    agent.trade(env, start_step=1, verbose=True, email=True)