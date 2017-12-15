from functools import wraps as _wraps
from itertools import chain as _chain
import json
from .utils import convert_to, Logger, dec_con
from decimal import Decimal
import pandas as pd
from time import sleep
from datetime import datetime, timezone, timedelta
import zmq
import threading
from multiprocessing import Process
from .exceptions import *
from cryptotrader.utils import send_email

debug = True

# Base classes
class ExchangeConnection(object):
    def __init__(self, period, pairs=[]):
        """
        :param tapi: exchange api instance: Exchange api instance
        :param period: int: Data period
        :param pairs: list: Pairs to trade
        """
        self.period = period
        self.pairs = pairs

    # Feed methods
    @property
    def balance(self):
        return NotImplementedError("This class is not intended to be used directly.")

    def returnBalances(self):
        return NotImplementedError("This class is not intended to be used directly.")

    def returnFeeInfo(self):
        return NotImplementedError("This class is not intended to be used directly.")

    def returnCurrencies(self):
        return NotImplementedError("This class is not intended to be used directly.")

    def returnChartData(self, currencyPair, period, start=None, end=None):
        return NotImplementedError("This class is not intended to be used directly.")

    # Trade execution methods
    def sell(self, currencyPair, rate, amount, orderType=False):
        return NotImplementedError("This class is not intended to be used directly.")

    def buy(self, currencyPair, rate, amount, orderType=False):
        return NotImplementedError("This class is not intended to be used directly.")

    def pair_reciprocal(self, df):
        df[['open', 'high', 'low', 'close']] = df.apply(
            {col: lambda x: str((Decimal('1') / convert_to.decimal(x)).quantize(Decimal('0E-8')))
             for col in ['open', 'low', 'high', 'close']}, raw=True).rename(columns={'low': 'high',
                                                                                     'high': 'low'}
                                                                            )
        return df.rename(columns={'quoteVolume': 'volume', 'volume': 'quoteVolume'})

## Feed daemon
# Server
class FeedDaemon(Process):
    """
    Data Feed server
    """
    def __init__(self, api={}, addr='ipc:///tmp/feed.ipc', n_workers=8, email={}):
        """

        :param api: dict: exchange name: api instance
        :param addr: str: client side address
        :param n_workers: int: n threads
        """
        super(FeedDaemon, self).__init__()
        self.api = api
        self.email = email
        self.context = zmq.Context()
        self.n_workers = n_workers
        self.addr = addr

        self.MINUTE, self.HOUR, self.DAY = 60, 60 * 60, 60 * 60 * 24
        self.WEEK, self.MONTH = self.DAY * 7, self.DAY * 30
        self.YEAR = self.DAY * 365

        self._nonce = int("{:.6f}".format(datetime.utcnow().timestamp()).replace('.', ''))

    @property
    def nonce(self):
        """ Increments the nonce"""
        self._nonce += 33
        return self._nonce

    def handle_req(self, req):

        req = req.split(' ')

        if req[0] == '' or len(req) == 1:
            return False

        elif len(req) == 2:
            return req[0], req[1]

        else:
            # Candle data
            if req[1] == 'returnChartData':

                if req[4] == 'None':
                    req[4] = datetime.utcnow().timestamp() - self.DAY
                if req[5] == 'None':
                    req[5] = datetime.utcnow().timestamp()

                call = (
                    req[0],
                    req[1],
                    {
                        'currencyPair': str(req[2]).upper(),
                        'period': str(req[3]),
                        'start': str(req[4]),
                        'end': str(req[5])
                        }
                    )
                return call

            if req[1] == 'returnTradeHistory':
                args = {'currencyPair': str(req[2]).upper()}
                if req[3] != 'None':
                    args['start'] = req[3]
                if req[4] != 'None':
                    args['end'] = req[4]

                return req[0], req[1], args

            # Buy and sell orders
            if req[1] == 'buy' or req[1] == 'sell':
                args = {
                    'currencyPair': str(req[2]).upper(),
                    'rate': str(req[3]),
                    'amount': str(req[4]),
                    }
                # order type specified?
                try:
                    possTypes = ['fillOrKill', 'immediateOrCancel', 'postOnly']
                    # check type
                    if not req[5] in possTypes:
                        raise ExchangeError('Invalid orderType')
                    args[req[5]] = 1
                except IndexError:
                    pass

                return req[0], req[1], args

    def worker(self):
        # Init socket
        sock = self.context.socket(zmq.REP)
        sock.connect("inproc://workers.inproc")

        while True:
            try:
                # Wait for request
                req = sock.recv_string()

                Logger.info(FeedDaemon.worker, req)

                # Handle request
                call = self.handle_req(req)

                # Send request to api
                if call:
                    try:
                        self.api[call[0]].nonce = self.nonce
                        rep = self.api[call[0]].__call__(*call[1:])
                    except ExchangeError as e:
                        rep = e.__str__()
                        Logger.error(FeedDaemon.worker, "Exchange error: %s\n%s" % (req, rep))

                    except DataFeedException as e:
                        rep = e.__str__()
                        Logger.error(FeedDaemon.worker,  "DataFeedException: %s\n%s" % (req, rep))

                    if debug:
                        Logger.debug(FeedDaemon.worker, "Debug: %s" % req)

                    # send reply back to client
                    sock.send_json(rep)
                else:
                    raise TypeError("Bad call format.")

            except Exception as e:
                send_email(self.email, "DataFeed Error", e)
                sock.close()
                raise e

    def run(self):
        try:
            Logger.info(FeedDaemon, "Starting Feed Daemon...")

            # Socket to talk to clients
            clients = self.context.socket(zmq.ROUTER)
            clients.bind(self.addr)

            # Socket to talk to workers
            workers = self.context.socket(zmq.DEALER)
            workers.bind("inproc://workers.inproc")

            # Launch pool of worker threads
            for i in range(self.n_workers):
                thread = threading.Thread(target=self.worker, args=())
                thread.start()

            Logger.info(FeedDaemon.run, "Feed Daemon running. Serving on %s" % self.addr)

            zmq.proxy(clients, workers)

        except KeyboardInterrupt:
            clients.close()
            workers.close()
            self.context.term()

# Client
class DataFeed(ExchangeConnection):
    """
    Data feeder for backtesting with TradingEnvironment.
    """
    # TODO WRITE TESTS
    retryDelays = [2 ** i for i in range(5)]

    def __init__(self, period, pairs=[], exchange='', addr='ipc:///tmp/feed.ipc', timeout=20):
        """

        :param period: int: Data sampling period
        :param pairs: list: Pair symbols to trade
        :param exchange: str: FeedDaemon exchange to query
        :param addr: str: Client socked address
        :param timeout: int:
        """
        super(DataFeed, self).__init__(period, pairs)

        # Sock objects
        self.context = zmq.Context()
        self.addr = addr
        self.exchange=exchange
        self.timeout = timeout * 1000

        self.sock = self.context.socket(zmq.REQ)
        self.sock.connect(addr)

        self.poll = zmq.Poller()
        self.poll.register(self.sock, zmq.POLLIN)

    def __del__(self):
        self.sock.close()

    # Retry decorator
    def retry(func):
        """ Retry decorator """

        @_wraps(func)
        def retrying(*args, **kwargs):
            problems = []
            for delay in _chain(DataFeed.retryDelays, [None]):
                try:
                    # attempt call
                    return func(*args, **kwargs)

                # we need to try again
                except DataFeedException as problem:
                    problems.append(problem)
                    if delay is None:
                        Logger.debug(DataFeed, problems)
                        raise MaxRetriesException('retryDelays exhausted ' + str(problem))
                    else:
                        # log exception and wait
                        Logger.debug(DataFeed, problem)
                        Logger.error(DataFeed, "No reply... -- delaying for %ds" % delay)
                        sleep(delay)

        return retrying

    def get_response(self, req):

        req = self.exchange + ' ' + req

        # Send request
        try:
            self.sock.send_string(req)
        except zmq.ZMQError as e:
            if 'Operation cannot be accomplished in current state' == e.__str__():
                # If request timeout, restart socket
                Logger.error(DataFeed.get_response, "%s request timeout." % req)
                # Socket is confused. Close and remove it.
                self.sock.setsockopt(zmq.LINGER, 0)
                self.sock.close()
                self.poll.unregister(self.sock)

                # Create new connection
                self.sock = self.context.socket(zmq.REQ)
                self.sock.connect(self.addr)
                self.poll.register(self.sock, zmq.POLLIN)

                raise DataFeedException("Socket error. Restarting connection...")

        # Get response
        socks = dict(self.poll.poll(self.timeout))
        if socks.get(self.sock) == zmq.POLLIN:
            # If response, return
            return self.sock.recv_json()

        else:
            # If request timeout, restart socket
            Logger.error(DataFeed.get_response, "%s request timeout." % req)
            # Socket is confused. Close and remove it.
            self.sock.setsockopt(zmq.LINGER, 0)
            self.sock.close()
            self.poll.unregister(self.sock)

            # Create new connection
            self.sock = self.context.socket(zmq.REQ)
            self.sock.connect(self.addr)
            self.poll.register(self.sock, zmq.POLLIN)

            raise RequestTimeoutException("%s request timedout" % req)

    @retry
    def returnTicker(self):
        try:
            rep = self.get_response('returnTicker')
            assert isinstance(rep, dict)
            return rep

        except AssertionError:
            raise UnexpectedResponseException("Unexpected response from DataFeed.returnTicker")

    @retry
    def returnBalances(self):
        """
        Return balance from exchange. API KEYS NEEDED!
        :return: list:
        """
        try:
            rep = self.get_response('returnBalances')
            assert isinstance(rep, dict)
            return rep

        except AssertionError:
            raise UnexpectedResponseException("Unexpected response from DataFeed.returnBalances")

    @retry
    def returnFeeInfo(self):
        """
        Returns exchange fee informartion
        :return:
        """
        try:
            rep = self.get_response('returnFeeInfo')
            assert isinstance(rep, dict)
            return rep

        except AssertionError:
            raise UnexpectedResponseException("Unexpected response from DataFeed.returnFeeInfo")

    @retry
    def returnCurrencies(self):
        """
        Return exchange currency pairs
        :return: list:
        """
        try:
            rep = self.get_response('returnCurrencies')
            assert isinstance(rep, dict)
            return rep

        except AssertionError:
            raise UnexpectedResponseException("Unexpected response from DataFeed.returnCurrencies")

    @retry
    def returnChartData(self, currencyPair, period, start=None, end=None):
        """
        Return pair OHLC data
        :param currencyPair: str: Desired pair str
        :param period: int: Candle period. Must be in [300, 900, 1800, 7200, 14400, 86400]
        :param start: str: UNIX timestamp to start from
        :param end:  str: UNIX timestamp to end returned data
        :return: list: List containing desired asset data in "records" format
        """
        try:
            call = "returnChartData %s %s %s %s" % (str(currencyPair),
                                                    str(period),
                                                    str(start),
                                                    str(end))
            rep = self.get_response(call)

            if 'Invalid currency pair.' in rep:
                try:
                    symbols = currencyPair.split('_')
                    pair = symbols[1] + '_' + symbols[0]

                    call = "returnChartData %s %s %s %s" % (str(pair),
                                                            str(period),
                                                            str(start),
                                                            str(end))

                    rep =  json.loads(
                        self.pair_reciprocal(pd.DataFrame.from_records(self.get_response(call))).to_json(
                            orient='records'))
                except Exception as e:
                    raise e

            assert isinstance(rep, list), "returnChartData reply is not list"
            assert int(rep[-1]['date']), "Bad returnChartData reply data"
            assert float(rep[-1]['open']), "Bad returnChartData reply data"
            assert float(rep[-1]['close']), "Bad returnChartData reply data"
            return rep

        except AssertionError:
            raise UnexpectedResponseException("Unexpected response from DataFeed.returnChartData")

    @retry
    def returnTradeHistory(self, currencyPair='all', start=None, end=None):
        try:
            call = "returnTradeHistory %s %s %s" % (str(currencyPair),
                                                        str(start),
                                                        str(end))

            rep = self.get_response(call)

            assert isinstance(rep, dict)
            return rep
        except AssertionError:
            raise UnexpectedResponseException("Unexpected response from DataFeed.returnTradeHistory")

    @retry
    def sell(self, currencyPair, rate, amount, orderType=False):
        try:
            call = "sell %s %s %s %s" % (str(currencyPair),
                                                    str(rate),
                                                    str(amount),
                                                    str(orderType))
            rep = self.get_response(call)

            if 'Invalid currency pair.' in rep:
                try:
                    symbols = currencyPair.split('_')
                    pair = symbols[1] + '_' + symbols[0]

                    call = "sell %s %s %s %s" % (str(pair),
                                                 str(rate),
                                                 str(amount),
                                                 str(orderType))

                    rep = self.get_response(call)

                except Exception as e:
                    raise e

            assert isinstance(rep, str) or isinstance(rep, dict)
            return rep

        except AssertionError:
            raise UnexpectedResponseException("Unexpected response from DataFeed.sell")

    @retry
    def buy(self, currencyPair, rate, amount, orderType=False):
        try:
            call = "buy %s %s %s %s" % (str(currencyPair),
                                         str(rate),
                                         str(amount),
                                         str(orderType))
            rep = self.get_response(call)

            if 'Invalid currency pair.' in rep:
                try:
                    symbols = currencyPair.split('_')
                    pair = symbols[1] + '_' + symbols[0]

                    call = "buy %s %s %s %s" % (str(pair),
                                                 str(rate),
                                                 str(amount),
                                                 str(orderType))

                    rep = self.get_response(call)

                except Exception as e:
                    raise e

            assert isinstance(rep, str) or isinstance(rep, dict)
            return rep

        except AssertionError:
            raise UnexpectedResponseException("Unexpected response from DataFeed.buy")


# Test datafeeds
class BacktestDataFeed(ExchangeConnection):
    """
    Data feeder for backtesting with TradingEnvironment.
    """
    # TODO WRITE TESTS
    def __init__(self, tapi, period, pairs=[], balance={}, load_dir=None):
        super().__init__(period, pairs)
        self.tapi = tapi
        self.ohlc_data = {}
        self._balance = balance
        self.data_length = 0
        self.load_dir = load_dir
        self.tax = {'makerFee': '0.00150000',
                'nextTier': '600.00000000',
                'takerFee': '0.00250000',
                'thirtyDayVolume': '0.00000000'}

    def returnBalances(self):
        return self._balance

    def set_tax(self, tax):
        """
        {'makerFee': '0.00150000',
                'nextTier': '600.00000000',
                'takerFee': '0.00250000',
                'thirtyDayVolume': '0.00000000'}
        :param dict:
        :return:
        """
        self.tax.update(tax)

    def returnFeeInfo(self):
        return self.tax

    def returnCurrencies(self):
        if self.load_dir:
            try:
                with open(self.load_dir + '/currencies.json') as file:
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

        index = pd.date_range(start=start,
                              end=end,
                              freq="%dT" % self.period).ceil("%dT" % self.period)

        for pair in self.pairs:
            ohlc_df = pd.DataFrame.from_records(
                            self.tapi.returnChartData(
                            pair,
                            period=self.period * 60,
                            start=start,
                            end=end
                ),
                nrows=index.shape[0]
            )

            i = -1
            last_close = ohlc_df.at[ohlc_df.index[i], 'close']
            while not dec_con.create_decimal(last_close).is_finite():
                i -= 1
                last_close = ohlc_df.at[ohlc_df.index[i], 'close']
            # Replace missing values with last close
            fill_dict = {col: last_close for col in ['open', 'high', 'low', 'close']}
            fill_dict.update({'volume': '0E-16'})

            self.ohlc_data[pair] = ohlc_df.fillna(fill_dict).ffill()

        for key in self.ohlc_data:
            if not self.data_length or self.ohlc_data[key].shape[0] < self.data_length:
                self.data_length = self.ohlc_data[key].shape[0]

        for key in self.ohlc_data:
            if self.ohlc_data[key].shape[0] != self.data_length:
                # self.ohlc_data[key] = pd.DataFrame.from_records(
                #     self.tapi.returnChartData(key, period=self.period * 60,
                #     start=self.ohlc_data[key].date.iloc[-self.data_length],
                #     end=end
                #     ),
                # nrows=index.shape[0]
                # )
                self.ohlc_data[key] = self.ohlc_data[key].iloc[:self.data_length]

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
            self.ohlc_data[item].to_json(dir+'/'+str(item)+'_'+str(self.period)+'min.json', orient='records')

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
            self.ohlc_data[key] = pd.read_json(self.load_dir + dir +'/'+str(key)+'_'+str(self.period)+'min.json', convert_dates=False,
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
                raise ExchangeError("%d invalid candle period" % period)
            elif "Invalid pair" == e:
                raise ExchangeError("Invalid currency pair.")


class PaperTradingDataFeed(ExchangeConnection):
    """
    Data feeder for paper trading with TradingEnvironment.
    """
    # TODO WRITE TESTS
    def __init__(self, tapi, period, pairs=[], balance={}):
        super().__init__(period, pairs)
        self.tapi = tapi
        self._balance = balance

    def returnBalances(self):
        return self._balance

    def returnFeeInfo(self):
        return {'makerFee': '0.00150000',
                'nextTier': '600.00000000',
                'takerFee': '0.00250000',
                'thirtyDayVolume': '0.00000000'}

    def returnTicker(self):
        return self.tapi.returnTicker()

    def returnCurrencies(self):
        """
        Return exchange currency pairs
        :return: list:
        """
        return self.tapi.returnCurrencies()

    def returnChartData(self, currencyPair, period, start=None, end=None):
        """
        Return pair OHLC data
        :param currencyPair: str: Desired pair str
        :param period: int: Candle period. Must be in [300, 900, 1800, 7200, 14400, 86400]
        :param start: str: UNIX timestamp to start from
        :param end:  str: UNIX timestamp to end returned data
        :return: list: List containing desired asset data in "records" format
        """
        try:
            return self.tapi.returnChartData(currencyPair, period, start=start, end=end)
        except ExchangeError as error:
            if 'Invalid currency pair.' == error.__str__():
                try:
                    symbols = currencyPair.split('_')
                    pair = symbols[1] + '_' + symbols[0]
                    return json.loads(
                        self.pair_reciprocal(pd.DataFrame.from_records(self.tapi.returnChartData(pair, period,
                                                                                                 start=start,
                                                                                                 end=end
                                                                                                 ))).to_json(
                            orient='records'))
                except Exception as e:
                    raise e
            else:
                raise error


# Live datafeeds
class PoloniexConnection(DataFeed):
    def __init__(self, period, pairs=[], exchange='', addr='ipc:///tmp/feed.ipc', timeout=20):
        """
        :param tapi: exchange api instance: Exchange api instance
        :param period: int: Data period
        :param pairs: list: Pairs to trade
        """
        super().__init__(period, pairs, exchange, addr, timeout)

    @DataFeed.retry
    def returnChartData(self, currencyPair, period, start=None, end=None):
        """
        Return pair OHLC data
        :param currencyPair: str: Desired pair str
        :param period: int: Candle period. Must be in [300, 900, 1800, 7200, 14400, 86400]
        :param start: str: UNIX timestamp to start from
        :param end:  str: UNIX timestamp to end returned data
        :return: list: List containing desired asset data in "records" format
        """
        try:
            call = "returnChartData %s %s %s %s" % (str(currencyPair),
                                                    str(period),
                                                    str(start),
                                                    str(end))
            rep = self.get_response(call)

            if 'Invalid currency pair.' in rep:
                try:
                    symbols = currencyPair.split('_')
                    pair = symbols[1] + '_' + symbols[0]

                    call = "returnChartData %s %s %s %s" % (str(pair),
                                                            str(period),
                                                            str(start),
                                                            str(end))

                    rep =  json.loads(
                        self.pair_reciprocal(
                            pd.DataFrame.from_records(
                                self.get_response(call)
                            )
                        ).to_json(orient='records'))

                except Exception as e:
                    raise e

            assert isinstance(rep, list), "returnChartData reply is not list: %s" % str(rep)
            assert int(rep[-1]['date']), "Bad returnChartData reply data"
            assert float(rep[-1]['open']), "Bad returnChartData reply data"
            assert float(rep[-1]['close']), "Bad returnChartData reply data"
            return rep

        except AssertionError:
            raise UnexpectedResponseException("Unexpected response from DataFeed.returnChartData")
