try:
	from setuptools import setup
except ImportError:
	from distutils.core import setup

config = {
    'description': "Cryptotrader cryptotrader for automated cryptocurrency arbitrage",
	'author': "Fernando H'.' Canteruccio, José Olimpio de Almeida",
	'url': "www.github.com/naripok",
	'download_url': "www.github.com/naripok",
	'author_email': "fernando.canteruccio@gmail.com, jose.mendes13@hotmail.com",
	'version': "0.1",
	'install_requires': [
                         'pytest,'
                         'hypothesis',
                         'numpy',
                         'pandas',
                         'pymongo',
                         'matplotlib',
                         'bokeh',
                         'chainer',
                         'chainerrl'
                         'tensorflow'
                         'keras',
                         'kerasrl',
                         'empyrical'
                         ],
	'packages': ['cryptotrader'],
	'scripts': [],
	'name': "cryptotrader"
	}

setup(**config)