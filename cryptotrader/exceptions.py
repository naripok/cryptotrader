from requests.exceptions import RequestException

class ExchangeError(Exception):
    """ Exception for handling exchange api errors """
    pass

class DataFeedException(Exception):
    pass

class RequestTimeoutException(DataFeedException):
    pass

class MaxRetriesException(DataFeedException):
    pass

class UnexpectedResponseException(DataFeedException):
    pass