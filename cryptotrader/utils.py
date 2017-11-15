import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, DivisionByZero
from functools import partialmethod

import zmq
import msgpack

import numpy as np
# from bson import Decimal128
import math

decimal_cases = 1E-8

# Helper functions and classes
class Logger(object):
    logger = None

    @staticmethod
    def __init__(name, output_dir=None):
        """
        Initialise the logger
        """
        Logger.logger = logging.getLogger(name)
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


def safe_div(x, y, eps=Decimal('1E-8')):
    try:
        out = x / y
    except DivisionByZero:
        out = x / (y + eps)
    except InvalidOperation:
        out = x / (y + eps)

    return out


def floor_datetime(t, period):
    t -= timedelta(minutes=t.minute % period,
                      seconds=t.second,
                      microseconds=t.microsecond)
    return t


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
        return b / (b.sum() + decimal_cases)
    else:
        b = np.exp(x)
        return b / (b.sum() + decimal_cases)


def array_normalize(x, float=True):
    out = convert_to.decimal(x)

    try:
        out /= out.sum()
    except DivisionByZero:
        out /= (out.sum() + convert_to.decimal('1e-8'))
    except InvalidOperation:
        out /= (out.sum() + convert_to.decimal('1e-8'))

    out[-1] += convert_to.decimal('1.00000000') - out.sum()

    if float:
        return np.float32(out)
    else:
        return out


def simplex_proj(y):
    """ Projection of y onto simplex. """
    m = len(y)
    bget = False

    s = sorted(y, reverse=True)
    tmpsum = 0.

    for ii in range(m - 1):
        tmpsum = tmpsum + s[ii]
        tmax = (tmpsum - 1) / (ii + 1)
        if tmax >= s[ii + 1]:
            bget = True
            break

    if not bget:
        tmax = (tmpsum + s[m - 1] - 1) / m

    return np.maximum(y - tmax, 0.)


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


class convert_to(object):
    _quantizer = Decimal('0E-8')
    _quantize = partialmethod(Decimal.quantize, _quantizer)
    _convert_array = np.vectorize(Decimal)
    _quantize_array = np.vectorize(lambda x: Decimal(x).quantize(convert_to._quantizer))

    # @staticmethod
    # def decimal128(data):
    #     if isinstance(data, np.float32):
    #         data = np.float64(data)
    #     return Decimal128(convert_to._quantize(Decimal(data)))

    @staticmethod
    def decimal(data):
        try:
            return Decimal(data).quantize(convert_to._quantizer)
        except TypeError:
            if isinstance(data, np.ndarray):
                # shape = data.shape
                # output = np.empty(data.flatten().shape, dtype=np.dtype(Decimal))
                # for i, item in enumerate(data.flatten()):
                #     output[i] = Decimal(np.float64(item)).quantize(convert_to._quantizer)
                # return output.reshape(shape)
                return convert_to._quantize_array(data.astype(str))
            else:
                return Decimal.from_float(np.float64(data)).quantize(convert_to._quantizer)
        except InvalidOperation:
            if abs(data) > Decimal('1e15'):
                raise InvalidOperation("Numeric overflow in convert_to.decimal")
            elif data == np.nan or math.nan:
                raise InvalidOperation("NaN encountered in convert_to.decimal")
        except Exception as e:
            print(data)
            print(e)
            raise e


# ZMQ sockets helpers
def write(_socket, msg, flags=0, block=True):
    if block:
        _socket.send(msgpack.packb(msg), flags=flags)
        return True
    else:
        try:
            _socket.send(msgpack.packb(msg), flags=flags | zmq.NOBLOCK)
            return True
        except zmq.Again:
            return False


def read(_socket, flags=0, block=True):
    if block:
        return msgpack.unpackb(_socket.recv(flags=flags))
    else:
        try:
            return msgpack.unpackb(_socket.recv(flags=flags | zmq.NOBLOCK))
        except zmq.Again:
            return False


def send_array(socket, A, flags=0, copy=False, track=False, block=True):
    """send a numpy array with metadata"""
    md = dict(
        dtype=str(A.dtype),
        shape=A.shape,
    )
    if block:
        socket.send_json(md, flags | zmq.SNDMORE)
        return socket.send(A, flags, copy=copy, track=track)
    else:
        try:
            socket.send_json(md, flags | zmq.SNDMORE | zmq.NOBLOCK)
            return socket.send(A, flags| zmq.NOBLOCK, copy=copy, track=track)
        except zmq.Again:
            return False


def recv_array(socket, flags=0, copy=False, track=False, block=True):
    """recv a numpy array"""
    if block:
        md = socket.recv_json(flags=flags)
        msg = socket.recv(flags=flags, copy=copy, track=track)
        buf = bytearray(msg)
        A = np.frombuffer(buf, dtype=md['dtype'])
        return A.reshape(md['shape'])
    else:
        try:
            md = socket.recv_json(flags=flags | zmq.NOBLOCK)
            msg = socket.recv(flags=flags | zmq.NOBLOCK, copy=copy, track=track)
            buf = bytearray(msg)
            A = np.frombuffer(buf, dtype=md['dtype'])
            return A.reshape(md['shape'])
        except zmq.Again:
            return False
