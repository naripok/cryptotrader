import logging
from datetime import datetime
from decimal import Decimal
from functools import partialmethod

import numpy as np
from bson import Decimal128


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


def array_softmax(x, SAFETY=2.0):
    """
    Compute softmax values for each sets of scores in x.

    Rows are scores for each class.
    Columns are predictions (samples).
    """
    # scoreMatExp = np.exp(np.asarray(x))
    # return scoreMatExp / scoreMatExp.sum()

    mrn = np.finfo(x.dtype).max # largest representable number
    thr = np.log(mrn / x.size) - SAFETY
    amx = x.max()
    if(amx > thr):
        b = np.exp(x - (amx-thr))
        return b / b.sum()
    else:
        b = np.exp(x)
        return b / b.sum()


def array_normalize(x, SAFETY=2.0):
    # out = convert_to.decimal(x)
    # print(out)
    # if out.sum() == convert_to.decimal('0.0'):
    #     out = out / (out.sum() + convert_to.decimal('1e-12'))
    # else:
    #     out = out / out.sum()
    #
    # if out.sum() > convert_to.decimal('1.0'):
    #     out[-1] += convert_to.decimal('1.0') - out.sum()
    # if out.sum() < convert_to.decimal('1.0'):
    #     out[-1] += convert_to.decimal('1.0') - out.sum()
    # print(out)
    # return np.float32(out)

    x = np.array(x, dtype=np.float64)
    mrn = np.finfo(x.dtype).max  # largest representable number
    thr = np.log(mrn / x.size) - SAFETY
    amx = x.max()
    if (amx > thr):
        b = x - (amx - thr)
        return b / (b.sum() + 1e-12)
    else:
        b = x
        return b / (b.sum() + 1e-12)


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