import gym

import numpy as np
import time

import keras

class Gym():

    def __init__(self, envname,
                     iterations=1,
                     max_steps=None,
                     action_mode='argmax',
                     result_mode='avg'):

        ## Init environment
        self.env = gym.make(envname)
        
        self.iterations = iterations
        self.max_steps = max_steps

        '''
        ACTION MODE:
            Determines how output data from neural network
            will be treated. Three options:
                - raw
                - argmax
                - tanh
        '''
        self.action_mode = action_mode

        self.result_mode = result_mode

    def evaluate(self, 
                specimen,
                model,
                render=False,
                verbose=False,
                time_sleep=.0):
        
        if specimen != None:
            ## Load specimen
            model.set_weights(specimen)

        reward_log = [] ## For later use in total reward sum if iterations > 1 
        for iters in range(self.iterations):

            ## Reset environment 
            state = self.env.reset()

            t_reward = 0 ## Reset total reward
            
            if self.max_steps != None:
                ## Finite time steps
                for step in range(self.max_steps):
                    ## Render env
                    if render:
                        self.env.render()
                    ## Format state
                    state = np.array([state])
                    ## Pass forward state data 
                    output = model.predict(state) 

                    ## Format output to use it as next action
                    if self.action_mode == 'argmax':
                        action = np.argmax(output[0])

                    elif self.action_mode == 'raw':
                        action = output[0]

                    elif self.action_mode == 'tanh':
                        action = np.tanh(output[0])

                    ## Run new step
                    state, reward, done, _ = self.env.step(action)
                    time.sleep(time_sleep) ## Wait time

                    ## Add current reard to total
                    t_reward += reward
                    
                    if done:
                        break

                ## Used if iterations > 1
                reward_log.append(t_reward)

            else:
                ## Test agent until game over
                done = False
                while not done:
                    ## Render env
                    if render:
                        self.env.render()

                    ## Format state
                    state = np.array([state])
                    ## Pass forward state data 
                    output = model.predict(state) 

                    ## Format output to use it as next action
                    if self.action_mode == 'argmax':
                        action = np.argmax(output[0])

                    elif self.action_mode == 'raw':
                        action = output[0]

                    elif self.action_mode == 'tanh':
                        action = np.tanh(output[0])

                    ## Run new step
                    state, reward, done, _ = self.env.step(action)
                    time.sleep(time_sleep) ## Wait time

                    ## Add current reard to total
                    t_reward += reward
                    ## End game if game over
                    if done:
                        break

                ## Used if iterations > 1
                reward_log.append(t_reward)

                ## Verbose
                if verbose:
                    print('Total reward: ', t_reward) 
                
        if self.result_mode == 'sum':
            ## Sum of total rewards in all iterations
            return sum(reward_log)

        elif self.result_mode == 'avg':
            ## Average of total rewards from all iterations 
            return np.mean(reward_log)

