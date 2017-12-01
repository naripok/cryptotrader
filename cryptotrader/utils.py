import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, DivisionByZero, getcontext, Context
from functools import partialmethod
import zmq
import msgpack

import numpy as np
# from bson import Decimal128
import math

# Decimal precision
getcontext().prec = 64
getcontext().Emax = 33
getcontext().Emin = -33
dec_con = getcontext()

# Decimal constants
dec_zero = dec_con.create_decimal('0E-16')
dec_one = dec_con.create_decimal('1.0000000000000000')
dec_eps = dec_con.create_decimal('1E-16')
dec_qua = dec_con.create_decimal('1E-8')

# Decimal vector operations
dec_vec_div = np.vectorize(dec_con.divide)
dec_vec_mul = np.vectorize(dec_con.multiply)
dec_vec_sub = np.vectorize(dec_con.subtract)

# Reward decimal context
rew_con = Context(prec=64, Emax=33, Emin=-33)
rew_quant = rew_con.create_decimal('0E-32')

# Debug flag
debug = True

# logger
class Logger(object):
    logger = logging.getLogger(__name__)

    @staticmethod
    def __init__(name, output_dir=None):
        """
        Initialise the logger
        """
        # Logger.logger = logging.getLogger(name)
        Logger.logger.setLevel(logging.DEBUG)
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
        Write error log
        :param method: Method name
        :param str: Log message
        """
        Logger.logger.error('[%s]\n%s\n' % (method, str))

    @staticmethod
    def debug(method, str):
        """
        Write debug log
        :param method: Method name
        :param str: Log message
        """
        Logger.logger.debug('[%s]\n%s\n' % (method, str))


# Helper functions and classes
def safe_div(x, y, eps=dec_eps):
    try:
        out = dec_con.divide(x, y)
    except DivisionByZero:
        out = dec_con.divide(x, y + eps)
    except InvalidOperation:
        out = dec_con.divide(x, y + eps)
    except TypeError:
        try:
            out = x / y
        except DivisionByZero:
            out = x / (y + eps)
        except InvalidOperation:
            out = x / (y + eps)

    return out


def floor_datetime(t, period):
    if period > 60:
        hours = t.hour % 2
    else:
        hours = 0

    t -= timedelta(
        hours=hours,
        minutes=t.minute % period,
        seconds=t.second,
        microseconds=t.microsecond)
    return t


# Array methods
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
        return b / (b.sum() + 1e-16)
    else:
        b = np.exp(x)
        return b / (b.sum() + 1e-16)


def array_normalize(x, float=True):
    out = convert_to.decimal(x)

    try:
        out /= out.sum()
    except DivisionByZero:
        out /= (out.sum() + dec_eps)
    except InvalidOperation:
        out /= (out.sum() + dec_eps)

    out[-1] += dec_con.create_decimal('1.00000000') - out.sum()

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


def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w


class convert_to(object):
    _quantizer = dec_zero
    _quantize = partialmethod(Decimal.quantize, _quantizer)
    _convert_array = np.vectorize(dec_con.create_decimal)
    _quantize_array = np.vectorize(lambda x: dec_con.create_decimal(x).quantize(convert_to._quantizer))

    # @staticmethod
    # def decimal128(data):
    #     if isinstance(data, np.float32):
    #         data = np.float64(data)
    #     return Decimal128(convert_to._quantize(Decimal(data)))

    @staticmethod
    def decimal(data):
        try:
            return dec_con.create_decimal(data).quantize(convert_to._quantizer)
        except TypeError:
            if isinstance(data, np.ndarray):
                return convert_to._quantize_array(data.astype(str))
            else:
                return Decimal.from_float(np.float64(data)).quantize(convert_to._quantizer)
        except InvalidOperation:
            if abs(data) > Decimal('1e20'):
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
