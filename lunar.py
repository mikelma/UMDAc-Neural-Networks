import numpy as np
import matplotlib.pyplot as plt

import keras

from keras.models import Model
from keras.layers import Input, Dense

from UMDAc.UMDAc import UMDAc
from UMDAc.Wrappers.gym import GYM

GENERATIONS = 500
GEN_SIZE = 200
SURV = .5
RAND_SURV = .3 

NOISE = None 
SEED = None
MAX_STEPS = 400
ITERATIONS = 3

problem = GYM('LunarLander-v2',
              iterations=ITERATIONS,
              max_steps=MAX_STEPS)

a = Input(shape=(8,))
b = Dense(4)(a)

model = Model(inputs=a, outputs=b)

umdac = UMDAc(model,
             problem=problem,
             gen_size=GEN_SIZE)

for generation in range(GENERATIONS):

    history = umdac.train(surv=SURV, 
                rand_surv=RAND_SURV,
                noise=NOISE)

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

    if max(history['max']) == history['max'][-1]:

        names = list(umdac.fitness.keys())
        f = list(umdac.fitness.values())

        best = umdac.gen[names[f.index(max(f))]]
        umdac.save_specimen(best, 'lunar_RAND.h5')
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

t_r = problem.evaluate(best, model, 
                        render=True,
                        verbose=True)
print('Average reward of 100 iterations: ', t_r)
