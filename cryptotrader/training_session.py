import gc
gc.collect()
import pymongo as pm
import numpy as np
np.random.seed(42)
from driver import get_historical, Apocalipse, convert_to
from agents import clear, SaveOnInterval, TerminateOnNaN, EIIE, ConstrainedOrnsteinUhlenbeckProcess

client = pm.MongoClient(host='25.50.224.109', port=27017, connect=True)

db = client.db

# Get data
freq = 15

obs_steps = 50

data_dir = '../historical_data/'

files = [
    data_dir + 'bitstampUSD_1-min_data_2012-01-01_to_2017-05-31.csv',
    data_dir + 'krakenUSD_1-min_data_2014-01-07_to_2017-05-31.csv',
    data_dir + 'coinbaseUSD_1-min_data_2014-12-01_to_2017-05-31.csv'
]

dfs = []
for file in files:
    dfs.append(get_historical(start='2017-01-01 00:00:00', end='2017-04-30 00:00:00', freq=freq, file=file))

## ENVIRONMENT INITIALIZATION
env = Apocalipse(db, 'EIIE_0.2_train_session')
# Set environment options
env.set_freq(freq)
env.set_obs_steps(obs_steps)

# Add data tables
env.add_table(name='exch_bitstamp_btcusd_trades')
env.add_table(name='exch_kraken_ltcusd_trades')
env.add_table(name='exch_kraken_xrpusd_trades')

# Add backtest data
for i, key in enumerate(env.tables.keys()):
    env.add_df(df=dfs[i], symbol=key)
del dfs

for symbol in env.tables.keys():
    env.add_symbol(symbol)
    env.set_init_crypto(0.0, symbol)
    env.set_tax(0.0025, symbol)
env.set_init_fiat(500.0)

# Clean pools
env._reset_status()
env.clear_dfs()

env.set_training_stage(True)
env.set_observation_space()
env.set_action_space()
obs = env.reset(reset_funds=True, reset_results=True)

model = EIIE(env, nb_steps_warmup_critic=10000, nb_steps_warmup_actor=10000, lr=3e-5, clipnorm=1., batch_size=50,
                 gamma=.99, target_model_update=1e-3, theta=4., mu=0., sigma=5., sigma_min=0., n_steps_annealing=10000,
                 name='EIIE_0.2')

model.load()
model.load_memory()
print("Actor network:")
model.actor.summary()
print("Critic network:")
model.critic.summary()

soi = SaveOnInterval(model, verbose=1, period_memory=10000, period_weights=1000)
ton = TerminateOnNaN()

model.env.reset(reset_funds=True, reset_results=True)

history = model.fit(model.env, nb_steps=int(1e9), nb_max_episode_steps=30, action_repetition=1, callbacks=[soi, ton], verbose=1,
            visualize=True, nb_max_start_steps=0, start_step_policy=None, log_interval=10000)

model.save()
model.save_memory()
print("Training session status:0", model.env.status)
print("Training session done!")