from requests.exceptions import RequestException

class ExchangeError(Exception):
    """ Exception for handling exchange api errors """
    pass

class RetryException(ExchangeError):
    """ Exception for retry decorator """
    pass

class DataFeedException(Exception):
    pass

class DataFeedRetryException(DataFeedException):
    pass
