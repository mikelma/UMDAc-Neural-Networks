import matplotlib.pyplot as plt
import numpy as np

import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

from UMDAc.UMDAc import UMDAc
from UMDAc.Wrappers.Gym import Gym

### HYPERPARAMETERS ###

GENERATIONS = 1000
GEN_SIZE = 100
SURV = .5 
RAND_SURV = .3 

NOISE = None 
FILENAME = 'car_result.h5' # Filename of best specimen

ITERATIONS = 1
MAX_STEPS = 200

## Initialize Gym problem 
problem = Gym('CarRacing-v0',
              iterations=ITERATIONS,
              max_steps=MAX_STEPS,
              action_mode='raw')

in_shape = (96, 96, 3) 
action_size = problem.env.action_space.shape[0]

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
             problem=problem,
             gen_size=GEN_SIZE)

### TRAINING ###
for generation in range(GENERATIONS):

    ## Train
    history = umdac.train(surv=SURV, 
                rand_surv=RAND_SURV,
                noise=NOISE)

    ## Generation's average total reward
    avg_f = history['avg'][-1]

    print(generation, ' / ', GENERATIONS,' avg reward: ', avg_f)

    ## Plot training info online
    plt.clf()

    plt.plot(range(generation + 1), history['min'],
             label='Minimum')
    plt.plot(range(generation + 1), history['max'],
             label='Maximum')
    plt.plot(range(generation + 1), history['avg'], 
             label='Average')
    plt.grid(b=True, which='major', color='#DDDDDD', 
             linestyle='-')
    plt.legend()
    plt.xlabel('Generation')
    plt.ylabel('Reward')

    plt.draw()
    plt.pause(.00001)
    
    ## Save best specimen to .h5 file 
    if max(history['max']) == history['max'][-1]:

        names = list(umdac.fitness.keys())
        f = list(umdac.fitness.values())

        best = umdac.gen[names[f.index(max(f))]]
        umdac.save_specimen(best, FILENAME)
        print('Best specimen saved')

## Plot graph
plt.show()

## Render best speciemens
print('')
print('-'*5, ' Rendering best specimen ', '-'*5)

## Select best specimen 
names = list(umdac.fitness.keys())
f = list(umdac.fitness.values())

best = umdac.gen[names[f.index(max(f))]]

problem.iterations = 100

## Render best specimen
t_r = problem.evaluate(best, model, 
                        render=True,
                        verbose=True)

print('Average reward of 100 iterations: ', t_r)

