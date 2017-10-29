# Cryptotrader - Cryptocurrency Trader Agent and Agent Development Environment

This repository contains infra-structure for agent backtesting and cryptocoin trading. 
It is under active development and is provided "AS-IS", without any warrants whatsoever.
## Getting Started
### Prerequisites

In order to get started, you will need python and some dependencies:

```
Python 3.2+
numpy
pandas
bokeh
chainer
optunity
ta-lib
empyrical
jupyter
```

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