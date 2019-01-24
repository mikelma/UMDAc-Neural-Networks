from os import listdir
import os
from os.path import isfile, join

## HYPERPARAMETERS
FORMAT = '.h5'
MAX_STEPS = None
SEED = 0
RENDER = True

## List files
mypath = os.getcwd() ## Current path
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

## Find valid files
fs = []
for f in files:
    if FORMAT in f: 
        fs.append(f)
## If no valid files
print('')
if len(fs) == 0:
    print('No .npy files found.')
    quit()

## Show valid files
print('Select file to load:')
print('')
for i, f in enumerate(fs):
    print('  [',i, '] ', f)

## Make choice
print('')
sel = None 
while sel not in range(len(fs)):
    sel = int(input('Selection > '))

sname = fs[sel] ## Selected filename

## List of available environments
envs = ['CartPole-v0','LunarLander-v2', 
        'BipedalWalker-v2']

print('\n'+'Select environment:')
## Show envs
for i, envname in enumerate(envs):
    print('['+str(i)+'] '+envname)

## Select environment
print('')
sel = None 
while sel not in range(len(envs)):
    sel = int(input('Selection > '))

## Init env
import gym
env = gym.make(envs[sel]) 

## Init UMDAc
from UMDAc.UMDAc import UMDAc

umdac = UMDAc(model=None, 
             gen_size=1,
             max_steps=MAX_STEPS,
             env=env,
             seed=SEED) 

## Evaluate specimen, render enabled

l = []
f = []

umdac.load_specimen(sname)

tr = umdac.gym_evaluate(None, RENDER)
print('\n', 'total reward: ', tr, '\n')
