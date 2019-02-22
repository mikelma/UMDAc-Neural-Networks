#!/usr/bin/env python3

# Univariate Marginal Distribution Algorithm

import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import time

from keras.models import model_from_json
import keras

class UMDAc():

    def __init__(self, 
                 model, 
                 problem, 
                 gen_size): 

        np.random.seed(None) 

        ## Global variables
        self.model = model
        self.gen_size = gen_size
        
        self.problem = problem

        if self.model != None:

            self.fitness = {} # Init fitness log
            
            ## Create first generation randomly
            self.gen = {} # Init generation 0

            ## Create random specimens
            for i in range(gen_size):
                ## Generate specimen weights and biases
                specimen = []
                for layer in model.get_weights():
                    specimen.append(np.random.uniform(
                        -1,1,layer.shape))                            
                self.gen['s'+str(i)] = np.array(specimen)

        self.fitness = {}
        
        ## Initialize training fitness logger
        self.history = {
            'avg':[],
            'min':[],
            'max':[]}
        
    def train(self, 
              surv, 
              rand_surv=None, 
              selection_mode='max', 
              noise=None):

        n_surv = int(self.gen_size*surv)

        if rand_surv == None:
            n_random_surv = 0
        else:
            n_random_surv = int(n_surv*rand_surv)

        n_surv -= n_random_surv

        survivors = []
        survivors_fitness = []
        
        ## Evaluate population
        for name in self.gen:
            survivors.append(name)

            specimen = self.gen[name]
            t_reward = self.problem.evaluate(specimen, 
                                             self.model) 
            self.fitness[name] = t_reward

        ## Collect data about generation
        survivors = list(self.fitness.keys()) ## Survivors' names
        survivors_fitness = list(self.fitness.values()) ## Survivors's fitnesses

        self.history['avg'].append(np.mean(survivors_fitness))
        self.history['min'].append(min(survivors_fitness))
        self.history['max'].append(max(survivors_fitness))

        worsts = [] ## Worst specimens names
        worsts_fitness = [] ## Worst specimens fitness values

        ## Select best fitness survivors
        n_r = len(survivors) - n_surv ## Number of not survivor specimens 
        for n in range(n_r):
            
            ## Select worst specimen
            if selection_mode == 'max':
                indx = survivors_fitness.index(min(survivors_fitness))

            elif selection_mode == 'min':
                indx = survivors_fitness.index(max(survivors_fitness))

            ## Save worsts 
            worsts.append(survivors[indx])    
            worsts_fitness.append(survivors_fitness[indx])
            ## Delete worsts from survivors lists
            del survivors[indx]
            del survivors_fitness[indx]
        
        if rand_surv != None: 
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

        new_names = list(self.new.keys())

        ## Replace worst specimens for new ones
        for i, name in enumerate(worsts):
            # print(name)
            self.gen[name] = copy.deepcopy(self.new[new_names[i]]) 

        return self.history

    def save_specimen(self, specimen, filename='specimen.h5'):

        from keras.models import load_model
        ## Load specimen's weights to model        
        self.model.set_weights(specimen)
        ## Save model and specimen's weights
        self.model.save(filename)  

    def load_model(self, filename='specimen.h5'):

        del self.model  # delete the existing model

        from keras.models import load_model
        ## Load new model and weights 
        self.model = load_model(filename)

if __name__ == '__main__':

    import numpy as np

    import keras

    from keras.models import Model
    from keras.layers import Input, Dense

    from Wrappers.Gym import Gym

    cartpole = Gym('CartPole-v0')

    GENERATIONS = 15
    GEN_SIZE = 30
    SURV = .5
    RAND_SURV = None 

    NOISE = .1 

    a = Input(shape=(4,))
    b = Dense(2)(a)

    model = Model(inputs=a, outputs=b)

    umdac = UMDAc(model,
                 problem=cartpole,
                 gen_size=GEN_SIZE)

    for generation in range(GENERATIONS):

        history = umdac.train(surv=SURV, 
                    rand_surv=RAND_SURV,
                    noise=NOISE)

        avg_f = history['avg'][-1]

        print(generation, ' / ', GENERATIONS,' avg reward: ', avg_f)

    # quit()
    ## Render best speciemens
    print('')
    print('-'*5, ' Rendering best specimen ', '-'*5)

    ## Select best specimen 
    names = list(umdac.fitness.keys())
    f = list(umdac.fitness.values())

    best = umdac.gen[names[f.index(max(f))]]

    cartpole.iterations = 100

    t_r = cartpole.evaluate(best, model, 
                            render=True,
                            verbose=True)

    print('Total reward: ', t_r)

