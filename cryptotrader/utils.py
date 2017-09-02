from bson import Decimal128
from functools import partialmethod
from decimal import Decimal
import logging
from datetime import datetime
import numpy as np
import pandas as pd


def array_softmax(x):
    """
    Compute softmax values for each sets of scores in x.

    Rows are scores for each class.
    Columns are predictions (samples).
    """
    scoreMatExp = np.exp(np.asarray(x))
    return scoreMatExp / scoreMatExp.sum()


def array_normalize(x):

    out = convert_to.decimal(x)

    if out.sum() == convert_to.decimal('1.0'):
        out = out / (out.sum() + convert_to.decimal('1e-12'))
    else:
        out = out / out.sum()

    if out.sum() > convert_to.decimal('1.0'):
        out[-1] += convert_to.decimal('1.0') - out.sum()
    if out.sum() < convert_to.decimal('1.0'):
        out[-1] += convert_to.decimal('1.0') - out.sum()

    return np.float32(out)


# Helper functions and classes
class convert_to(object):
    _quantize = partialmethod(Decimal.quantize, Decimal('1e-12'))

    @staticmethod
    def decimal128(data):
        if isinstance(data, np.float32):
            data = np.float64(data)
        return Decimal128(convert_to._quantize(Decimal(data)))

    @staticmethod
    def decimal(data):
        if isinstance(data, np.float32) or isinstance(data,float):
            data = np.float64(data)
            return Decimal.from_float(data).quantize(Decimal('1e-12'))
        if isinstance(data, np.ndarray):
            output = []
            shape = data.shape
            for item in data.flatten():
                item = np.float64(item)
                output.append(Decimal(item).quantize(Decimal('1e-12')))
            return np.array(output).reshape(shape)
        else:
            return Decimal(data).quantize(Decimal('1e-12'))


class Logger(object):
    logger = None

    @staticmethod
    def __init__(name, output_dir=None):
        """
        Initialise the logger
        """
        Logger.logger = logging.getLogger('Cryptocoin arbiter agent logging file')
        Logger.logger.setLevel(logging.ERROR)
        Logger.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s \n%(message)s\n')

        if output_dir is None:
            slogger = logging.StreamHandler()
            slogger.setFormatter(formatter)
            Logger.logger.addHandler(slogger)

        else:
            flogger = logging.FileHandler("%s%s_%s.log" % (output_dir, name, datetime.now().strftime("%Y%m%d_%H%M%S")))
            flogger.setFormatter(formatter)
            Logger.logger.addHandler(flogger)

    @staticmethod
    def info(method, str):
        """
        Write info log
        :param method: Method name
        :param str: Log message
        """
        Logger.logger.info('[%s]\n%s\n' % (method, str))

    @staticmethod
    def error(method, str):
        """
        Write info log
        :param method: Method name
        :param str: Log message
        """
        Logger.logger.error('[%s]\n%s\n' % (method, str))


def get_historical(file, freq, start=None, end=None):
    """
    Gets historical data from csv file
    return sampled ohlc pandas dataframe
    """
    assert freq >= 1
    freq = "%dmin" % (freq)

    if isinstance(file, pd.core.frame.DataFrame):
        df = file
    else:
        df = pd.read_csv(file)
        df['Timestamp'] = pd.to_datetime(df.Timestamp, infer_datetime_format=True, unit='s')
        df.set_index('Timestamp', drop=True, inplace=True)
    if start:
        df = df.drop(df.loc[:start].index)
    if end:
        df = df.drop(df.loc[end:].index)
    try:
        df = df.drop(['Volume_(Currency)', 'Weighted_Price'], axis=1)
        df = df.rename({'Volume_(BTC)': 'volume'})
    except:
        pass
    df.columns = ['open', 'high', 'low', 'close', 'volume']

    df.ffill(inplace=True)
    df.fillna(1e-12, inplace=True)

    index = df.resample(freq).first().index
    out = pd.DataFrame(index=index)
    out['open'] = df.resample(freq).first().open
    out['high'] = df.resample(freq).max().high
    out['low'] = df.resample(freq).min().low
    out['close'] = df.resample(freq).last().close
    out['volume'] = df.resample(freq).sum().volume

    return out.applymap(convert_to.decimal)
