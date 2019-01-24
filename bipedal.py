from UMDAc.UMDAc import UMDAc
import matplotlib.pyplot as plt

import numpy as np
import gym

import keras

from keras.models import Model
from keras.layers import Input, Dense

### HYPERPARAMETERS ###
LOG = False
LOG_FILENAME = 'bipedal_log.txt'

GENERATIONS = 1000
GEN_SIZE = 300
N_SURV = 100 
N_RAND_SURV = 50 

ENV_NAME = 'BipedalWalker-v2'
AUTO_STOP = True
SOLVED = 600

ITERATIONS = 1
MAX_STEPS = 600*ITERATIONS

LOG_NOTES = 'gensize:'+str(GEN_SIZE)+' , nsurv:'+str(
    N_SURV)+' nrandsurv:'+str(N_RAND_SURV)
 
## Environment initialization
env = gym.make(ENV_NAME)

## Model initialization
a = Input(shape=(env.observation_space.shape[0],))
x = Dense(15, activation='relu')(a)
x = Dense(15, activation='relu')(x)
b = Dense(env.action_space.shape[0], activation='tanh')(x)

model = Model(inputs=a, outputs=b)

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
        t_reward = umdac.gym_evaluate(specimen, render=False)
        
        reward_log.append(t_reward)
        
        ## Update fitness value
        umdac.fitness[name] = t_reward
    
    ## Train, create new generation
    umdac.train(N_SURV, N_RAND_SURV)    
    
    ## Save best specimen
    names = list(umdac.fitness.keys())
    f = list(umdac.fitness.values())

    if len(max_rewards) > 0:
        if max(max_rewards) < max(f) or len(max_rewards) == 1:

            best = names[f.index(max(f))]
            umdac.save_specimen(umdac.gen[best], 'result_specimen')

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

    ## Stop if game is solved
    if AUTO_STOP and max(reward_log) >= SOLVED:
        break

umdac.env.close() ## Close environment 

## Save training data 
if LOG:
    f = open(LOG_FILENAME, 'w')
    f.write('Data order: avg, max, min. Notes: '+LOG_NOTES+'\n')
    f.write(str(avg_reward_log)+'\n')
    f.write(str((max_rewards))+'\n')
    f.write(str((min_rewards))+'\n')
    f.close()

print('Training finished!')

## Plot training data
plt.show()

## Select best specimen 
best = list(umdac.fitness.keys())[
list(umdac.fitness.values()).index(max(
umdac.fitness.values()))]

## Render best speciemens
print('')
print('-'*5, ' Rendering best specimen ', '-'*5)

umdac.iterations = 1

rlog = []
while 1:
    ## For each specimen
    specimen = umdac.gen[best]
    ## Tests specimen in environment
    t_reward = umdac.gym_evaluate(specimen,
                                 render=True)
    rlog.append(t_reward)
    avg = sum(rlog) / len(rlog)
    print('Total reward: ', t_reward, 
          ',  average total reward: ', avg) 
