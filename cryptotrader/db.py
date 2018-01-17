"""
Investors and funds database
"""

import numpy as np
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta
import pymongo as pm
from decimal import Decimal, localcontext, ROUND_UP, ROUND_DOWN
from multiprocessing import Process
import zmq
from .utils import Logger

class DBClient(object):
    def __init__(self, db, api, context, sock_addr="ipc:///tmp/db.ipc"):
        self.deposits = db.deposits
        self.withdrawals = db.withdrawals
        self.funds = db.funds
        self.totalfunds = db.totalfunds
        self.clients = db.clients
        self.api = api
        self.context = context
        self.sock_addr = sock_addr

    def calc_portval(self, quote="BTC"):
        with localcontext() as ctx:
            ctx.rounding = ROUND_UP
            balance = self.api.returnBalances()
            ticker = self.api.returnTicker()
            portval = Decimal('0.0')
            for symbol in balance:
                if quote + "_" + symbol in list(ticker.keys()):
                    portval += Decimal(balance[symbol]) * Decimal(ticker[quote + "_" + symbol]['highestBid'])
            portval += Decimal(balance[quote])
            return portval

    def calc_profit(self, prevval, start, end):
        with localcontext() as ctx:
            ctx.rounding = ROUND_DOWN

            try:
                deposits = pd.DataFrame.from_records(self.deposits.find({'date': {
                    '$gte': start,
                    '$lte': end
                },
                    'status': {'$eq': 'COMPLETE'}
                })).funds.apply(
                    lambda x: Decimal(x))

            except AttributeError:
                deposits = np.array([Decimal('0.0')])

            try:
                withdrawals = pd.DataFrame.from_records(self.withdrawals.find({'date': {
                    '$gte': start,
                    '$lte': end
                    }
                })).funds.apply(
                    lambda x: Decimal(x))
            except AttributeError:
                withdrawals = np.array([Decimal('0.0')])

            portval = self.calc_portval() - deposits.sum() + withdrawals.sum()

            return (portval - prevval) / prevval, deposits, withdrawals

    def discouted_profit(self, profit, fee='0.0025'):
        with localcontext() as ctx:
            ctx.rounding = ROUND_UP
            return profit - max(Decimal(fee) * Decimal(profit), Decimal('0E-8'))

    def add_client(self, name, email, btcwallet, address, phone, rg, cpf, date, fee, txid=None, funds=None,
                   currency=None):
        try:
            data = {'name': str(name),
                    'email': str(email),
                    'btcwallet': str(btcwallet),
                    'address': str(address),
                    'phone': str(phone),
                    'rg': str(rg),
                    'cpf': str(cpf),
                    'date': date,
                    'fee': str(fee)
                    }

            self.clients.insert_one(data)
            if funds:
                self.deposit(cpf, date, txid, funds, currency)

        except Exception as e:
            print("Something went wrong. Call the administrator!, error:", e)
            return e

    def pull_transaction_data(self, start=None, end=None):
        if not start:
            start = dt.utcnow() - timedelta(hours=2, minutes=10)
        if not end:
            end = dt.utcnow()
        # Pull exchange data
        return self.api.returnDepositsWithdrawals(start.timestamp(), end.timestamp()), start, end

    def write_funds(self, owner, date, funds):
        try:
            data = {
                'owner': str(owner),
                'date': date,
                'funds': str(funds)
            }
            self.funds.insert_one(data)

        except Exception as e:
            Logger.info(DBClient.write_funds, 'Error: %s' % str(e))
            return e
        return True

    def update_funds(self, date):
        try:
            prev_funds = pd.DataFrame.from_records(self.totalfunds.find().sort('date', pm.DESCENDING).limit(1))
            prevval = Decimal(prev_funds.funds[0])
            start = prev_funds.date.astype(dt).values[0]

            Logger.debug(DBClient.update_funds, prev_funds)

            self.update_deposits(*self.pull_transaction_data(start, date))

            profit, deposits, withdrawals = self.calc_profit(prevval, start, date)

            Logger.debug(DBClient.update_funds, (profit, deposits, withdrawals))

            for owner in self.clients.find():
                Logger.info(DBClient.update_deposits, "Updating %s funds..." % str(owner))
                try:
                    funds = Decimal(pd.DataFrame.from_records(
                        self.funds.find(
                            {
                                'date': {'$lte': date},
                                'owner': {'$eq': owner['cpf']}
                             }
                        ).sort('date', pm.DESCENDING).limit(1))['funds'][0])

                except IndexError:
                    funds = Decimal('0.0')

                Logger.info(DBClient.update_deposits, "Previous funds: %s" % str(funds))

                transactions = Decimal('0.0')
                try:
                     transactions += deposits[owner['name']]
                except IndexError:
                    Logger.info(DBClient.update_deposits, "No deposits found for %s" % str(owner['cpf']))
                try:
                     transactions -= withdrawals[owner['name']]
                except IndexError:
                    Logger.info(DBClient.update_deposits, "No withdrawals found for %s" % str(owner['cpf']))

                new_funds = funds * (1 + self.discouted_profit(profit, owner['fee'])) + transactions

                self.write_funds(owner['cpf'], date, new_funds)

                Logger.info(DBClient.update_deposits, "New funds: %s" % str(new_funds))

            self.update_totalfunds(date)

        except Exception as e:
            Logger.error(DBClient.update_funds, 'Error: %s' % str(e))

    def update_totalfunds(self, date):
        try:
            data = {
                'date': date,
                'funds': str(self.calc_portval())
            }
            self.totalfunds.insert_one(data)

        except Exception as e:
            Logger.error(DBClient.update_totalfunds, 'Error: %s' % str(e))
            return e
        return True

    def update_deposits(self, exchange_data, start, end):
        exchange_deposits = pd.DataFrame.from_records(exchange_data['deposits'])
        Logger.debug(DBClient.update_deposits, exchange_deposits)
        # Pull database data
        megali_deposits = pd.DataFrame.from_records(self.deposits.find({'date': {'$gte': start, '$lte': end}}))
        Logger.debug(DBClient.update_deposits, megali_deposits)
        # Update deposits status
        if megali_deposits.shape[0] > 0:
            for i, row in megali_deposits.loc[megali_deposits.status == 'PENDING'].iterrows():
                try:
                    Logger.info(DBClient.update_deposits, exchange_deposits.loc[exchange_deposits.txid == row.txid])
                    if exchange_deposits.loc[exchange_deposits.txid == row.txid].status[i] == 'COMPLETE':
                        #                     megali_deposits.loc[i, 'status'] = 'COMPLETE'
                        self.deposits.update_one(
                            {"_id": megali_deposits.loc[i, '_id']},
                            {"$set": {"status": "COMPLETE", 'date': dt.fromtimestamp(
                                exchange_deposits.loc[exchange_deposits.txid == row.txid].timestamp)}}
                        )
                except Exception as e:
                    Logger.error(DBClient.update_deposits, "Error: %s"% str(e))
                    return e

        return True

    def update_withdrawals(self, exchange_data, start, end):
        # TODO FINISH THIS
        exchange_withdrawals = pd.DataFrame.from_records(exchange_data['withdrawals'])
        megali_withdrawals = pd.DataFrame.from_records(self.withdrawals.find({'date': {'$gte': start, '$lte': end}}))

        # Update withdrawals status
        for i, row in megali_withdrawals.loc[megali_withdrawals.status == 'PENDING'].iterrows():
            try:
                if exchange_withdrawals.loc[exchange_withdrawals.txid == row.txid].status[0].split(':')[0] == 'COMPLETE':
                    self.withdrawals.update_one(
                        {"_id": megali_withdrawals.loc[i, '_id']},
                        {"$set": {"status": "COMPLETE"}}
                    )
            except Exception as e:
                print("Something went wrong. Call the administrator!, error:", e)
                return e

        return True

    def deposit(self, owner, date, txid, funds, currency, status='PENDING'):
        try:
            data = {
                'owner': str(owner),
                'txid': str(txid),
                'date': date,
                'funds': str(funds),
                'currency': str(currency),
                'status': str(status)
            }

            self.deposits.insert_one(data)
        except Exception as e:
            Logger.error(DBClient.deposit, "Error: %s" % str(e))

    def withdraw(self, owner, date, txid, funds, currency, status='COMPLETE'):
        try:
            data = {
                'owner': str(owner),
                'txid': str(txid),
                'date': date,
                'funds': str(funds),
                'currency': str(currency),
                'status': str(status)
            }

            self.withdrawals.insert_one(data)
        except Exception as e:
            print("Something went wrong. Call the administrator!, error:", e)

    def create_indexes(self):
        self.clients.create_index([("date", pm.DESCENDING)])
        self.funds.create_index([("date", pm.DESCENDING)])
        self.withdrawals.create_index([("date", pm.DESCENDING)])
        self.deposits.create_index([("date", pm.DESCENDING)])

    def report(self):
        return False

    def handle_req(self, req):
        try:
            Logger.info(DBClient.handle_req, '%s request received.' % req)
            if req == 'update':
                self.update_funds(dt.utcnow())
                Logger.info(DBClient.handle_req, '%s request processed.' % req)

                return "Done. Total funds: %s" % str(self.calc_portval())

        except Exception as e:
            Logger.info(DBClient.handle_req, 'Error: %s' % str(e))
            return e

    def run(self):
        try:
            self.sock = self.context.socket(zmq.REP)
            self.sock.bind(self.sock_addr)
            while True:
                rep = self.handle_req(self.sock.recv_string())
                self.sock.send_string(str(rep))
        except KeyboardInterrupt:
            self.sock.close()

        except Exception as e:
            self.sock.send_string(str(e))

        finally:
            self.sock.close()