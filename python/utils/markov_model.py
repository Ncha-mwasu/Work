import random
from functools import partial

import numpy as np
import pandas as pd
from numpy.linalg import matrix_power

import config as cf
from python.utils.timer import Stage

# rand_data = [random.choice(list(Stage)) for i in range(100)]


def energy_consumedd(x):
    """ Return the energy consumption of the stage """
    if x == Stage.SLEEPING:
        return 0
    if x == Stage.SENSING:
        return cf.SENSING_ENERGY
    if x == Stage.TRANSMITTING:
        return cf.E_FS
    if x == Stage.RECEIVING:
        return cf.E_ELEC
    if x == Stage.AGGREGATING:
        return cf.E_DA
    if x == Stage.IDLE:
        return cf.E_IDLE
    return 0


def generate_transition_matrix(data):
    """ Generate the Transition Matrix
    Uses a probabilistic method to generate transition matrix. Computes all transitions from one state
    to another then use the transition to generate transition Matrix
    This actually returns the prob matrix
    """

    _df = {}
    # Current State data
    _df['state'] = data[:-1]
    # Next state of each current state is the next data in the list
    _df ["next_state"] = data[1:]
    cleaned_data = pd.DataFrame(_df)
    transitions = {d: {} for d in list(Stage)}
    # Check for transitions between states and store count
    for i in transitions.keys():
        for j in transitions.keys():
            transitions[i][j] = cleaned_data[
                    (cleaned_data["state"] == i) & (cleaned_data["next_state"] == j)].shape[0]

    # Calculate the Probability of Transition based on data from transtions
    df = pd.DataFrame(transitions)
    for i in range(df.shape[0]):
        df.iloc[i] = df.iloc[i]/(df.iloc[i].sum() or 1)

    transition_matrix = df.values
    open('markov_energy.txt', 'w+').write(str(transition_matrix))
    return transition_matrix

class MarkovChain:
    """ The Markov Chain Model Predictor """

    def __init__(self):
        """ Initialize the necessary Variables
        data is a 1D array containing states in chronological order which they were transitioned to
        """

        # Store the transitions from a state to this dict
        self.transition_matrix = None

    def get_current_state_vector(self, current_state):
        """ Get the current state vector
        This is usually of the form [0, 0, 1, 0], where 1 indicates the current state, and other states
        are in 0 state
        """

        v_state = np.zeros([len(list(Stage))])
        v_state[current_state.value] = 1
        return v_state

    def next_state(self, current_state, time_step=1, transition_matrix=None):
        """
        Returns the state of the random variable at the next time
        instance.

        Parameters
        ----------
        current_state: str
            The current state of the system.
        """
        tmx = np.atleast_1d(transition_matrix)
       

        current_state_vector = self.get_current_state_vector(current_state)
        a = current_state_vector.dot(matrix_power(tmx, time_step))

        state  = list(Stage)[np.argmax(a)] 

        Er = energy_consumedd(state)
        #print("Er->", Er)
        
        B = np.array([0, cf.E_TRANSMITTING_PROCESSING, 0, 0, 0, 0,   
            
            cf.E_PROCESSING_TRANSMITTING, 0, 0, cf.E_PROCESSING_SENSING, 0, 0, 
           
             0, 0, 0, 0, 0, 0, 

             0, 0, 0, 0, 0, 0,      
            
             0, cf.E_RECEIVING_PROCESSING, 0, 0, 0, 0,      
             
             cf.E_IDLE_TRANSMITTING, cf.E_IDLE_PROCESSING, 0, cf.E_IDLE_SENSING, cf.E_IDLE_RECEIVING, 0
              ]).reshape((6,6))
        # print("EB before IS->")
        Ks = np.multiply(B, tmx)
        Eb = a.dot(Ks.dot(np.ones((6,1))))
       # print("EB IS->", Eb)
        return (Er+Eb)

    def predict(self, current_state, transition_matrix=None, no_predictions=1):
        """
        Generates the next states of the system.

        Parameters
        ----------
        current_state: str
            The state of the current random variable.

        no: int
            The number of future states to generate.
        """
        future_states = []        
        for i in range(0, no_predictions):
            next_state = (self.next_state(current_state, time_step=i+1, transition_matrix=transition_matrix))
            future_states.append((next_state))
        return (future_states)
