#!/usr/bin/env python3

# Univariate Marginal Distribution Algorithm

import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import time

import keras

class UMDAc():

    def __init__(self, model, gen_size, 
                 env, 
                 max_steps=None,
                 action_mode='raw',
                 seed=0, 
                 iterations=1, 
                 display_info=True):
        
        ## Global variables
        self.model = model
        self.gen_size = gen_size
        self.iterations = iterations
        
        self.seed = seed
        self.max_steps = max_steps

        self.env = env ## Environment

        '''
        ACTION MODE:
            Determines how output data from neural network
            will be treated. Three options:
                - raw
                - argmax
                - tanh
        '''
        self.action_mode = action_mode

        if display_info:
            ## Print environment info
            print('\n' + '#'*5, ' INFO  ' ,'#'*5)
            print('Observation space: ', self.env.observation_space)
            print('Iterations: ', self.iterations)
            print('')

        self.fitness = {} # Init fitness log
        
        ## Create first generation randomly
        self.gen = {} # Init generation 0

        ## Create random specimens
        for i in range(gen_size):
            ## Generate specimen weights and biases
            specimen = []
            for layer in model.get_weights():
                specimen.append(np.random.uniform(-1,1,layer.shape))                            
            self.gen['s'+str(i)] = np.array(specimen)

    def gym_evaluate(self, specimen,  
                    render=False, 
                    time_sleep=.0):

        ## Load specimen
        self.model.set_weights(specimen)

        seed = self.seed ## Initial random seed
        reward_log = [] ## For later use in total reward sum if iterations > 1 
        for iters in range(self.iterations):

            ## Reset environment 
            self.env.seed(seed)
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
                    output = self.model.predict(state) 

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
                ## Update seed to test agent in different scenarios
                seed += 1

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
                    output = self.model.predict(state) 

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
                seed += 1 ## Update random seed 
                
        ## Disable random seed
        ''' This prevents the algorithm to generate the
            same random numbers all time.   '''
        np.random.seed(None)
        ## Sum of total rewards in all iterations
        return sum(reward_log)

    def train(self, n_surv, n_random_surv, noise=None):
        
        ## Collect data about generation
        survivors = list(self.fitness.keys()) ## Survivors' names
        survivors_fitness = list(self.fitness.values()) ## Survivors's fitnesses

        worsts = [] ## Worst specimens names
        worsts_fitness = [] ## Worst specimens fitness values

        ## Select best fitness survivors
        n_r = len(survivors) - n_surv ## Number of not survivor specimens 
        for n in range(n_r):
            
            ## Select worst specimen
            indx = survivors_fitness.index(min(survivors_fitness))
            ## Save worsts 
            worsts.append(survivors[indx])    
            worsts_fitness.append(survivors_fitness[indx])
            ## Delete worsts from survivors lists
            del survivors[indx]
            del survivors_fitness[indx]

        ## Randomly select bad specimens to survive
        for i in range(n_random_surv):
            ## Random index
            indx = np.random.randint(len(worsts))
            ## Add random specimen to survivors 
            survivors.append(worsts[indx])
            survivors_fitness.append(worsts_fitness[indx])
            ## Update worst specimens' lists
            del worsts[indx]
            del worsts_fitness[indx]
        
        ## Generate new specimens (empty):
        self.new = {}
        for i in range(len(worsts)):
            self.new['n'+str(i)] = copy.deepcopy(self.gen['s0'])

        for i_l, layer in enumerate(self.gen['s0']):

            if len(layer.shape) == 1: 
                layer = layer.reshape((1, layer.shape[0]))
                isbias = True
            else:
                isbias = False

            for i in range(layer.shape[0]):
                for j in range(layer.shape[1]):
                    ## layer[i][j] weight of each survivor 
                    w = []
                    ## For each survivor
                    for name in survivors:
                        if isbias:
                            w.append(self.gen[name][i_l][j])
                        else:
                            w.append(self.gen[name][i_l][i][j])

                    if noise != None:
                        n_mut = int(len(w)*noise)
                        muts = np.random.rand(n_mut)

                        w = np.array(w)
                        np.random.shuffle(w)
                        
                        w = np.delete(w, range(len(w)-n_mut, len(w)), 0)

                        w = np.hstack((w, muts))
                        np.random.shuffle(w)
                    
                    ## Compute weights list's mean 
                    mean = np.mean(w)
                    ## Standard deviation
                    std = np.std(w)
                    
                    ## Get samples
                    samples = np.random.normal(mean, std, 
                                               len(worsts))
                    
                    i_sample = 0 ##  Iterator
                    ## Generate new specimens
                    for name in self.new:
                        ## Update weight  
                        if isbias:
                            self.new[name][
                                i_l][j] = samples[i_sample]
                        else:
                            self.new[name][
                                i_l][i][j] = samples[i_sample]
                        i_sample += 1 
        
        ## After generating a set of new specimens
        new_names = []
        new_fitness = [] 

        for name in self.new:
            ## Evaluate new specimens
            ## and store data for later comparison
            new_names.append(name)
            specimen = self.new[name]
            new_fitness.append(self.gym_evaluate(specimen))

        '''
        Selection. Replace all specimens in the worsts list
        with best specimens of the to_select lists.
        '''
        to_select_names = new_names+worsts
        to_select_fitness = new_fitness+worsts_fitness

        for i in range(len(worsts)):
            indx = np.argmax(to_select_fitness)
            
            ## Add selected specimen to new generation
            if 'n' in to_select_names[indx]:
                ## Replace specimen
                self.gen[worsts[i]] = copy.deepcopy(self.new[
                to_select_names[indx]])

            else:
                ## Replace specimen
                self.gen[worsts[i]] = copy.deepcopy(self.gen[
                to_select_names[indx]])

            ## Update selection lists
            del to_select_names[indx]
            del to_select_fitness[indx]

    def add_neurons(self, layer_name, n_neurons=1):

        ## To all specimens in generation
        for name in self.gen:

            ## Load specimen
            specimen = self.gen[name]

            last_indx = int(len(specimen) / 2) - 1 ## Number of layers
            sel_indx = int(layer_name[1]) ## Selected layer's index

            ## Add neuron to layer
            new_neuron = np.random.rand(
                specimen[layer_name].shape[0], n_neurons)
            specimen[layer_name] = np.hstack(
                (specimen[layer_name], new_neuron))
             
            ## Add new bias 
            new_bias = np.random.rand(1,n_neurons)
            specimen['b'+str(sel_indx)] = np.hstack(
                (specimen['b'+str(sel_indx)], new_bias))

            ## Check if the selected layer is 
            ## the last (output layer) of the net
            if sel_indx != last_indx:
                next_layer = specimen['h'+str(sel_indx+1)]
                ## Selected layer isn't the last 
                ## Generate new weights
                new_w = np.random.rand(n_neurons, next_layer.shape[1])
                ## Add weights to next layer
                specimen['h'+str(sel_indx+1)] = np.vstack(
                    (new_w, next_layer))

    def add_layer(self, n_neurons):
    ## Add one layer to all specimens
    ## The new layer is added before
    ## the output layer
        
        ## Define network's layers
        specimen = self.gen['s0']
        layers = []
        layers_shape = []
        biases = []
        biases_shape = []
        for l in specimen:
            if 'h' in l:
                layers.append(l)               
                layers_shape.append(specimen[l].shape)
            elif 'b' in l:
                biases.append(l)
                biases_shape.append(specimen[l].shape)

        for name in self.gen:
            ## Load specimen
            specimen = self.gen[name]
            ## Reset output layer 
            new_o = np.random.rand(n_neurons, self.action_size)
            ## Reset output layer bias 
            new_o_b = np.random.rand(1, self.action_size)
            ## Create new layer
            new_l = np.random.rand(layers_shape[-2][1], 
                                  n_neurons)
            new_l_b = np.random.rand(1, n_neurons)

            specimen[layers[-1]] = new_l 
            specimen[biases[-1]] = new_l_b
            specimen['h'+str(len(layers))] = new_o
            specimen['b'+str(len(biases))] = new_o_b
            

    def save_specimen(self, specimen, filename='specimen'):
        ## Save weights to .npy numpy file
        np.save(filename, specimen)

    def load_specimen(self, filename='specimen.npy'):
        ## Load specimen's weights from numpy .npy file 
        return np.load(filename)

if __name__ == '__main__':

    import gym
    import numpy as np

    ## Environment initialization
    env = gym.make('CartPole-v0')

    import keras

    from keras.models import Model
    from keras.layers import Input, Dense

    GENERATIONS = 20
    GEN_SIZE = 30
    N_SURV = 10
    N_RAND_SURV = 5 
    NOISE = None 

    a = Input(shape=(4,))
    b = Dense(2)(a)

    model = Model(inputs=a, outputs=b)

    umdac = UMDAc(model,
                 gen_size=GEN_SIZE,
                 action_mode='argmax',
                 max_steps=200,
                 env=env)

    # umdac.save_specimen(umdac.gen['s0'])
    # s = umdac.load_specimen()

    for generation in range(GENERATIONS):

        r_log = [] ## Total rewards log

        for name in umdac.gen:
            
            specimen = umdac.gen[name]
            t_reward = umdac.gym_evaluate(specimen, False)

            umdac.fitness[name] = t_reward
            r_log.append(t_reward)

        umdac.train(n_surv=N_SURV, n_random_surv=N_RAND_SURV,
                   noise=NOISE)

        avg_f = sum(r_log) / len(r_log)
        print(generation, ' / ', GENERATIONS,' avg reward: ', avg_f)

    ## Render best speciemens
    print('')
    print('-'*5, ' Rendering best specimen ', '-'*5)

    ## Select best specimen 
    names = list(umdac.fitness.keys())
    f = list(umdac.fitness.values())
    best = umdac.gen[names[f.index(max(f))]]

    while 1:
        t_r = umdac.gym_evaluate(best, True)
        print('Total reward: ', t_r)
        ## Set random seed to random value
        umdac.seed = np.random.randint(254)


