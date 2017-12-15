# Cryptotrader - Cryptocurrency Trader Agent and Agent Development Environment

THIS REPOSITORY IS IN DEVELOPMENT STAGE, SO THE PREREQUISITES, DOCUMENTATION AND NOTEBOOKS ARE OFTEN OUTDATED!
IF YOU MISS SOME PRE REQUISITE, IT IS PROBABLY MISSING, SO JUST INSTALL IT BY HAND.
CONTRIBUTIONS ARE VERY WELCOME!

This repository contains infra-structure for agent backtesting and cryptocoin trading.
Currently it supports backtesting and papertrading using poloniex's historical dada and also live trading at poloniex.
We are, however, not associated or affiliated anyhow with poloniex or any services that it provides.
It is under active development and is provided "AS-IS", without any warrants whatsoever.
## Getting Started
### Prerequisites

In order to get started, you will need python and some dependencies:

```
- Python 3.6
- numpy
- pandas
- bokeh
- chainer
- optunity
- ta-lib
- cvxopt
- empyrical
- jupyter
```

You also will need ta-lib binaries installed on your system.

### Installing

Just clone this repository and install using pip:
```
git clone git@github.com:naripok/cryptotrader.git
cd cryptotrader/
pip3 install .
```

### Running backtests
Inside notebooks directory you will find jupyter notebooks containing code to optimize and backtest some example agents. 
You just need to run it under an active internet connection, as the data will be downloaded on the go.

### Running Paper Trading
In order to run paper trading, you will need to specify the initial balance.
The api is similar to the backtest one. You can also specify some gmail addr and password in order to receive trade and debug logs.
There is, as well, an example notebook inside notebooks dir.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Buy us a coffee

If this module or some of our strategies makes you some money, we gladly accept some coffee... or coins... =D

BTC: 1Q9xCNhqs5gWToovR8petSRYMreEvZEQyA

Thank you very much!