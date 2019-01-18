from UMDAc.UMDAc import UMDAc
from activation.relu import relu
from activation.tanh import tanh

import numpy as np
import time

import gym

from ple.games.pong import Pong
from ple import PLE

NET_SIZE = [7] 
ACTIVATION = relu

MAX_STEPS = 800
ITERATIONS = 1
# ENV_NAME = 'BipedalWalker-v2'
env = Pong(250, 250)

## Environment initialization
# env = gym.make(ENV_NAME)

## Initialize UMDAc
umdac = UMDAc(1, NET_SIZE, ACTIVATION, 
              env, 
              max_steps=MAX_STEPS,
              action_mode='raw',
              iterations=ITERATIONS)

new = umdac.load_specimen('resultPLE.txt')

while 1:
    t_reward = umdac.ple_evaluate(new, time_sleep=0.05)
    print('Total reward: ', t_reward)
