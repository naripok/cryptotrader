"""
EIIE A3C training script

You must have an instance of FeedDaemon running in order to use this script.
If you want to use a direct connection for data, change the papi data feed instance.
"""

import sys
sys.path.insert(0, '../')

from os import listdir
import logging
logging.basicConfig(level=logging.ERROR)

import numpy as np
np.random.seed(42)
from time import time
from datetime import datetime, timedelta

from cryptotrader.datafeed import DataFeed
from cryptotrader.exchange_api.poloniex import Poloniex
from cryptotrader.envs.trading import BacktestDataFeed, BacktestEnvironment
from cryptotrader.envs.utils import make_balance
from cryptotrader.agents import cn_agents

import chainer as cn
from chainerrl.optimizers import rmsprop_async
from chainerrl import experiments
from chainerrl.agents import a3c
from chainerrl.experiments.hooks import LinearInterpolationHook
from chainerrl import misc
from cryptotrader.agents.cn_agents import phi, PrintProgress


def main():
    import argparse
    parser = argparse.ArgumentParser()

    # Simulation params
    parser.add_argument('--name', type=str, default='EIIE_A3C')
    parser.add_argument('--processes', type=int, default=4,
                        help="number of environment instances to use")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed")
    parser.add_argument('--obs_steps', type=int, default=64,
                        help="Observation steps, number of candles required by the agent for calculations")
    parser.add_argument('--period', type=int, default=30,
                        help="Observation period in minutes, also trading frequency")
    parser.add_argument('--max_episode_len', type=int, default=5,
                        help="Max timesteps per episode")
    parser.add_argument('--steps', type=int, default=1e6,
                        help="Training steps")
    parser.add_argument('--eval-interval', type=int, default=None)
    parser.add_argument('--eval-n-runs', type=int, default=10)

    # Learning params
    parser.add_argument('--n_filters_in', type=int, default=8,
                        help="number of input filters heads to train")
    parser.add_argument('--n_filters_out', type=int, default=256,
                        help="number of pattern recognition neurons to train")
    parser.add_argument('--t_max', type=int, default=5,
                        help="Timesteps before update main model")
    parser.add_argument('--grad_noise', type=float, default=0.0,
                        help="gradient noise to apply")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=1000,
                        help="learning rate linear decay rate")
    parser.add_argument('--alpha', type=float, default=0.99,
                        help="Exponential decay rate of the second order moment for rmsprop optimizer")
    parser.add_argument('--rmsprop-epsilon', type=float, default=1e-1,
                        help="fuzz factor")
    parser.add_argument('--clip_grad', type=float, default=100,
                        help="Clip gradient norm")
    parser.add_argument('--reward-scale-factor', type=float, default=1.,
                        help="scale environment reward")

    # Regularization
    parser.add_argument('--beta', type=float, default=1e-3,
                        help="entropy regularization weight for policy")
    parser.add_argument('--beta_decay', type=float, default=1000,
                        help="entropy regularization decay rate")
    parser.add_argument('--l2_reg', type=float, default=0,
                        help="l2 regularization coefficient")
    parser.add_argument('--l1_reg', type=float, default=0,
                        help="l1 regularization coefficient")
    parser.add_argument('--gamma', type=float, default=0.999,
                        help="discount factor")

    # Misc
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--monitor', action='store_true', default=False)
    parser.add_argument('--download', action='store_true', default=False)
    parser.add_argument('--datafeed', action='store_true', default=False)

    # Dir options
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--out_dir', type=str, default='./save')
    parser.add_argument('--load_dir', type=str, default='./save')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--logger-level', type=int, default=logging.ERROR)

    args = parser.parse_args()
    logging.getLogger().setLevel(args.logger_level)
    if args.seed is not None:
        misc.set_random_seed(args.seed)

    args.outdir = experiments.prepare_output_dir(args, args.log_dir)

    # Simulation Params
    # Universe
    pairs = [
        'BTC_ETH',
        'BTC_BCH',
        'BTC_XRP',
        'BTC_STR',
        'BTC_LTC',
        'BTC_DASH',
        'BTC_XMR',
        'BTC_ETC',
        'BTC_ZEC',
        'BTC_BTS',
        'BTC_LSK',
        'BTC_XEM',
        'BTC_VTC',
        'BTC_STRAT',
        'BTC_EMC2',
        'BTC_NXT',
        'BTC_OMG'
    ]  # Universe, some survivor bias here...
    fiat_symbol = 'BTC'  # Quote symbol

    init_funds = make_balance(crypto=0.0, fiat=100.0, pairs=pairs)  # Initial equally distributed portfolio

    # NN params
    timesteps = args.obs_steps - 1
    n_filters_in = args.n_filters_in
    n_filters_out = args.n_filters_out

    if args.load_dir:
        try:
            last_save = np.argmax([int(d.split('_')[0]) for d in listdir(args.load_dir) if '.' not in d])
            load_dir = args.load_dir + "/" + listdir(args.load_dir)[last_save]
            global_t = int(listdir(args.load_dir)[int(last_save)].split('_')[0])
        except ValueError as e:
            load_dir=False
            global_t = 0
    else:
        load_dir = None
        global_t = 0

    # Make environment function
    if args.datafeed:
        papi = DataFeed(args.period, pairs, 'polo_test', 'ipc://feed.ipc')
    else:
        papi = BacktestDataFeed(Poloniex(), args.period, pairs)

    if args.download:
        print("Downloading data...")
        papi.download_data(end=datetime.timestamp(datetime.utcnow() - timedelta(days=50)),
                            start=datetime.timestamp(datetime.utcnow() - timedelta(days=300)))
        papi.save_data(args.data_dir + '/train')

    def make_env(process_idx, test):
        tapi = BacktestDataFeed(papi, args.period, pairs=pairs, balance=init_funds, load_dir=args.data_dir)
        tapi.load_data('/train')

        # Environment setup
        env = BacktestEnvironment(args.period, args.obs_steps, tapi, fiat_symbol, args.name)
        env.setup()

        if args.monitor and process_idx == 0:
            env = gym.wrappers.Monitor(env, args.outdir)

        if not test:
            misc.env_modifiers.make_reward_filtered(
                env, lambda x: x * args.reward_scale_factor)

        return env

    # Model declaration
    print("Instantiating model")
    model = cn_agents.A3CEIIE(timesteps, len(pairs) + 1, n_filters_in, n_filters_out)#.to_gpu(0)

    # Optimizer
    opt = rmsprop_async.RMSpropAsync(lr=args.lr, eps=args.rmsprop_epsilon, alpha=args.alpha)
    opt.setup(model)
    if args.clip_grad:
        opt.add_hook(cn.optimizer.GradientClipping(args.clip_grad))
    if args.grad_noise:
        opt.add_hook(cn.optimizer.GradientNoise(args.grad_noise))
    if args.l2_reg:
        opt.add_hook(cn.optimizer.WeightDecay(args.l2_reg))
    if args.l1_reg:
        opt.add_hook(cn.optimizer.Lasso(args.l1_reg))

    # Agent
    print("Building agent")
    agent = a3c.A3C(model,
                    opt,
                    t_max=args.t_max,
                    gamma=args.gamma,
                    beta=args.beta,
                    phi=phi,
                    normalize_grad_by_t_max=True,
                    act_deterministically=False,
                    v_loss_coef=1.0)

    # Load information
    if load_dir:
        agent.load(load_dir)
        print("Model loaded from %s" % (load_dir))

    # Training hooks
    pp = PrintProgress(time())

    def lr_setter(env, agent, value):
        agent.optimizer.lr = value

    def beta_setter(env, agent, value):
        agent.beta = value

    lr_decay = LinearInterpolationHook(int(args.steps), args.lr, args.lr / args.lr_decay, lr_setter)
    beta_decay = LinearInterpolationHook(int(3 * args.steps / 4), args.beta, args.beta / args.beta_decay, beta_setter)

    # Training session
    try:
        print("Training starting...\n")
        with np.errstate(divide='ignore'):
            experiments.train_agent_async(
                agent=agent,
                outdir=args.out_dir,
                processes=args.processes,
                make_env=make_env,
                profile=args.profile,
                steps=int(args.steps),
                eval_n_runs=args.eval_n_runs,
                eval_interval=args.eval_interval,
                max_episode_len=args.max_episode_len,
                global_step_hooks=[pp, lr_decay, beta_decay],
                resume_step=global_t
                )
    except KeyboardInterrupt:
        print("\nThx for the visit. Good bye.")


if __name__ == '__main__':
    main()
