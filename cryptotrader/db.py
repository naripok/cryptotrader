"""
Investors and funds database
"""

import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
import pymongo as pm
from decimal import Decimal, localcontext, ROUND_UP, ROUND_DOWN
from .utils import send_email
from .utils import Logger
from time import sleep
from .utils import floor_datetime

class DBClient(object):
    def __init__(self, db, api, email, period):
        self.deposits = db.deposits
        self.withdrawals = db.withdrawals
        self.funds = db.funds
        self.totalfunds = db.totalfunds
        self.clients = db.clients
        self.api = api
        self.email = email
        self.period = period

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
                }))
                deposits_sum = deposits.funds.apply(lambda x: Decimal(x)).sum()
            except AttributeError:
                deposits = pd.DataFrame(np.array([Decimal('0.0')]), columns=['funds'])
                deposits_sum = Decimal('0.0')

            try:
                withdrawals = pd.DataFrame.from_records(self.withdrawals.find({'date': {
                    '$gte': start,
                    '$lte': end
                    }
                }))
                withdrawals_sum = withdrawals.funds.apply(lambda x: Decimal(x)).sum()
            except AttributeError:
                withdrawals = pd.DataFrame(np.array([Decimal('0.0')]), columns=['funds'])
                withdrawals_sum = Decimal('0.0')

            portval = self.calc_portval() - deposits_sum + withdrawals_sum

            Logger.info(DBClient.calc_profit, (deposits, withdrawals, portval))

            with localcontext() as ctx:
                ctx.rounding = ROUND_DOWN
                return (portval - prevval) / prevval, deposits, withdrawals

    def discouted_profit(self, profit, fee='0.0025'):
        with localcontext() as ctx:
            ctx.rounding = ROUND_UP
            tax = max(Decimal(fee) * Decimal(profit), Decimal('0E-8'))
            ctx.rounding = ROUND_DOWN
            return profit - tax

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
            self.write_funds(str(cpf), date, '0.0')
            if funds:
                self.deposit(cpf, date, txid, funds, currency)

        except Exception as e:
            Logger.error(DBClient.add_client, self.parse_error(e))
            return e

    def pull_transaction_data(self, start=None, end=None):
        if not start:
            start = datetime.utcnow() - timedelta(hours=4)
        if not end:
            end = datetime.utcnow()
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
            Logger.error(DBClient.write_funds, self.parse_error(e))
            return e
        return True

    def update_funds(self, date):
        try:
            # Get previous funds values
            prev_funds = pd.DataFrame.from_records(self.totalfunds.find().sort('date', pm.DESCENDING).limit(1))
            prevval = Decimal(prev_funds.funds[0])
            start = prev_funds.date.astype(datetime).values[0]
            Logger.debug(DBClient.update_funds, prev_funds)

            # Update pending transactions
            self.update_deposits(*self.pull_transaction_data(start - timedelta(hours=4), date))

            # Calculate step profit
            profit, deposits, withdrawals = self.calc_profit(prevval, start, date)
            Logger.debug(DBClient.update_funds, (profit, deposits, withdrawals))

            # For each client, update funds value
            for owner in self.clients.find():
                Logger.info(DBClient.update_deposits, "Updating %s funds..." % str(owner))
                try:
                    # find client last funds entry
                    funds = Decimal(pd.DataFrame.from_records(
                        self.funds.find(
                            {
                                'date': {'$lte': date},
                                'owner': {'$eq': str(owner['cpf'])}
                             }
                        ).sort('date', pm.DESCENDING).limit(1))['funds'][0])
                except IndexError:
                    # If not found, then there is no funds held
                    funds = Decimal('0.0')

                Logger.info(DBClient.update_deposits, "%s previous funds: %s" % (str(owner['cpf']), str(funds)))

                # Aggregate deposited funds
                transactions = Decimal('0.0')
                try:
                    with localcontext() as ctx:
                        ctx.rounding = ROUND_DOWN
                        d = Decimal(deposits[deposits.owner == owner['cpf']].funds.apply(lambda x: Decimal(x)).sum())
                        if not d.is_nan():
                            transactions += d
                except AttributeError as e:
                    Logger.info(DBClient.update_deposits, "No deposits found for %s" % str(owner['cpf']))

                # Aggregate withdrawn funds
                try:
                    with localcontext() as ctx:
                        ctx.rounding = ROUND_UP
                        w =  Decimal(withdrawals[withdrawals.owner == owner['cpf']].funds.apply(lambda x: Decimal(x)).sum())
                        if not w.is_nan():
                            transactions -= w
                except AttributeError:
                    Logger.info(DBClient.update_deposits, "No withdrawals found for %s" % str(owner['cpf']))

                # Calculate new hold value
                with localcontext() as ctx:
                    ctx.rounding = ROUND_DOWN
                    new_funds = funds * (1 + self.discouted_profit(profit, owner['fee'])) + transactions

                # Write to database
                self.write_funds(owner['cpf'], date, new_funds.quantize(Decimal('0E-8')))
                Logger.info(DBClient.update_deposits, "%s new funds: %s" % (str(owner['cpf']), str(new_funds)))

            # Calculate total liability funds, just for logging
            df_funds = pd.DataFrame.from_records(self.funds.find())

            account_funds = df_funds[df_funds.date == df_funds.iloc[-1].date].funds.apply(
                lambda x: Decimal(x).quantize(Decimal('0E-8'))).sum()

            Logger.info(DBClient.update_deposits, "Total liability funds: %s" % str(account_funds))

            # Update account funds on database
            self.update_totalfunds(date)

        except Exception as e:
            Logger.error(DBClient.update_funds, self.parse_error(e))

    def update_totalfunds(self, date):
        try:
            funds = self.calc_portval()

            data = {
                'date': date,
                'funds': str(funds)
            }

            Logger.info(DBClient.update_totalfunds, "Total asset funds: %s" % str(funds))

            self.totalfunds.insert_one(data)

        except Exception as e:
            Logger.error(DBClient.update_totalfunds, self.parse_error(e))
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
                    Logger.debug(DBClient.update_deposits, exchange_deposits)
                    Logger.debug(DBClient.update_deposits, row.txid)
                    Logger.info(DBClient.update_deposits, exchange_deposits.loc[exchange_deposits.txid == row.txid])
                    if exchange_deposits.loc[exchange_deposits.txid == row.txid].status.all() == 'COMPLETE':
                        self.deposits.update_one(
                            {"_id": megali_deposits.loc[i, '_id']},
                            {"$set": {"status": "COMPLETE", }}
                        )
                except Exception as e:
                    Logger.error(DBClient.update_deposits, self.parse_error(e))
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
                Logger.error(DBClient.update_withdrawals, self.parse_error(e))
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
            Logger.error(DBClient.deposit, self.parse_error(e))

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
            Logger.error(DBClient.withdraw, self.parse_error(e))

    def create_indexes(self):
        self.clients.create_index([("date", pm.DESCENDING)])
        self.funds.create_index([("date", pm.DESCENDING)])
        self.withdrawals.create_index([("date", pm.DESCENDING)])
        self.deposits.create_index([("date", pm.DESCENDING)])

    def report(self, profits, funds, date):

        msg = "DataBase Report:\n"
        msg += "date: %s" % str(date)
        msg += ""

        return False

    def parse_error(self, e, *args):
        error_msg = '\n' + ' error -> ' + type(e).__name__ + ' in line ' + str(
            e.__traceback__.tb_lineno) + ': ' + str(e)

        for args in args:
            error_msg += "\n" + str(args)

        return error_msg

    def run(self):
        last_action_time = floor_datetime(datetime.utcnow(), self.period) - timedelta(minutes=3)
        Logger.info(DBClient.run, "DB Start time: %s" % str(last_action_time))
        can_act = True
        while True:
            try:
                # Log action time
                loop_time = datetime.utcnow()
                Logger.info(DBClient.run, "DB Looptime: %s" % str(loop_time))
                # Can act?
                if loop_time >= last_action_time + timedelta(minutes=self.period):
                    can_act = True

                # If can act, run strategy and step environment
                if can_act:
                    # Log action time
                    last_action_time = floor_datetime(loop_time + timedelta(minutes=10), self.period) - timedelta(minutes=3)

                    Logger.info(DBClient.run, "DB Last action time: %s" % str(last_action_time))

                    # Run deposit and withdrawals verification and update client and total funds
                    self.update_funds(loop_time)

                    # self.report()

                    Logger.info(DBClient.run, "Total Funds: %s" % str(self.calc_portval()))

                    can_act = False

                else:
                    sleep(60 * 2)

            except KeyboardInterrupt:
                break
