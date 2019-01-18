from UMDAc.UMDAc import UMDAc
from activation.relu import relu
import matplotlib.pyplot as plt

import numpy as np
from ple.games.flappybird import FlappyBird
from ple.games.pong import Pong
from ple.games.waterworld import WaterWorld
from ple.games.snake import Snake
from ple import PLE

### HYPERPARAMETERS ###
LOG = False
LOG_FILENAME = 'ple0.txt'

NET_SIZE = [10] 
ACTIVATION = relu

GENERATIONS = 900
GEN_SIZE = 100
N_SURV = 35
N_RAND_SURV = 15 

ITERATIONS = 1
MAX_STEPS = None

LOG_NOTES = 'gensize:', str(GEN_SIZE),' , nsurv:', str(
    N_SURV), ' nrandsurv:', str(N_RAND_SURV)
 
## Environment initialization
#env = FlappyBird()
#env = Snake()
# env = WaterWorld(250, 250)
env = Pong(150, 150)
## Initialize UMDAc
umdac = UMDAc(GEN_SIZE, NET_SIZE, ACTIVATION, 
              env, 
              max_steps=MAX_STEPS,
              iterations=ITERATIONS, 
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
        t_reward = umdac.ple_evaluate(specimen)
        
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

    ## Save best specimen
    names = list(umdac.fitness.keys())
    f = list(umdac.fitness.values())
    b = names[f.index(max(f))] 
    umdac.save_specimen(umdac.gen[b], 'resultPLE.txt')

    ## Print some data during training
    print('generation ', i, '/', GENERATIONS,
          ', average reward: ', avg_reward)

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

while 1:
    ## For each specimen
    specimen = umdac.gen[best]
    ## Tests specimen in environment
    t_reward = umdac.ple_evaluate(specimen, time_sleep=.2)
    print('Total reward: ', t_reward) 
    ## Set random seed to random value
    umdac.seed = np.random.randint(254)
