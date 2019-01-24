from UMDAc.UMDAc import UMDAc
import matplotlib.pyplot as plt

import numpy as np
import gym

import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

### HYPERPARAMETERS ###

GENERATIONS = 1000
GEN_SIZE = 100
N_SURV = 30 
N_RAND_SURV = 20 

ITERATIONS = 1
MAX_STEPS = 200

## Environment initialization
env = gym.make('CarRacing-v0')

observation = env.reset()

in_shape = observation.shape
action_size = env.action_space.shape[0]

## Build Network
model = Sequential()
 
model.add(Convolution2D(5, kernel_size=(10,10), strides=(10,10), 
                        activation='relu',
                        data_format="channels_last", 
                        input_shape=in_shape))

model.add(MaxPooling2D(pool_size=(3,3)))
 
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(action_size, activation='tanh'))

model.summary()

## Initialize UMDAc
umdac = UMDAc(model,
              gen_size=GEN_SIZE, 
              env=env, 
              max_steps=MAX_STEPS,
              iterations=ITERATIONS, 
              action_mode='raw',
              display_info=True)

## Reset training data loggers    
avg_reward_log = []
max_rewards = []
min_rewards = []
last_avg_reward = 0

for i in range(GENERATIONS):    
    ## Reset reward logger
    reward_log = []
    for name in umdac.gen:
        ## Load specimen
        specimen = umdac.gen[name]
        ## Tests specimen in environment
        t_reward = umdac.gym_evaluate(specimen, render=True)
        
        reward_log.append(t_reward)
        
        ## Update fitness value
        umdac.fitness[name] = t_reward
    
    ## Train, create new generation
    umdac.train(N_SURV, N_RAND_SURV)    

    ## Calculate and log average reward
    avg_reward = sum(reward_log) / len(reward_log)
    avg_reward_log.append(avg_reward)
    ## Log max reward
    max_rewards.append(max(reward_log))
    min_rewards.append(min(reward_log))

    ## Plot training info online
    plt.clf()

    plt.plot(range(len(min_rewards)), min_rewards,
             label='Minimum')
    plt.plot(range(len(max_rewards)), max_rewards,
             label='Maximum')
    plt.plot(range(len(avg_reward_log)), avg_reward_log, 
             label='Average')
    plt.grid(b=True, which='major', color='#DDDDDD', 
             linestyle='-')
    plt.legend()
    plt.xlabel('Generation')
    plt.ylabel('Reward')

    plt.draw()
    plt.pause(.00001)

    ## Print some data during training
    print('generation ', i, '/', GENERATIONS,
          ', average reward: ', avg_reward)

