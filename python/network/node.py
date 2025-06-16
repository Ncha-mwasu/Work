import os
import sys

import time
from collections import OrderedDict
import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
from matplotlib.pyplot import xcorr
import config as cf
from python.network.energy_source import *
from python.utils import markov_model
from python.utils.timer import Stage, Timer
from python.utils.utils import *
from timer4 import SignalManager
class NoRunningFilter(logging.Filter):
    def filter(self, record):
        return False

class Controller(object):
    def __init__(self,id, ip, x,y,parent=None):
        self.pos_x = x
        self.pos_y = y
        self.energy_source = PluggedIn(self)
        self.scheduler = BackgroundScheduler
        self.id = id
        self.ip =  ip
        self.network_handler = parent
        self.alive = 1
        self.flow_table = None
        self.amount_transmitted = 0
        self.amount_received = 0
        self.is_sleeping = 0
        self.controller_membership = cf.BSID
        self.membership = -999
        self.next_hop = cf.BSID

    def is_head(self):
        return 0
    
    @classmethod
    def new_controller(cls, id, parent):
        """ Return a controller Node """
        node = cls(id, parent)
        node.energy_source = PluggedIn(cls)
        return node

    #Transmission for controller    
    def transmit(self, msg_length=None, destination=None, msg_type = 0):
        logging.debug("node %d transmitting." % (self.id))
        if not msg_length:
            msg_length = self.tx_queue_size
        msg_length += cf.HEADER_LENGTH
        if self.flow_table:
            destination = self.flow_table[self.id]["destination"]
            distance = self.distance_to_endpoint
        elif not destination:
            destination = self.network_handler[self.next_hop]
            distance = self.distance_to_endpoint
        else:
            distance = calculate_distance(self, destination)

        self.tx_queue_size = 0
        self.amount_transmitted += msg_length


    def receive(self, msg_length,msg_type = 0):
        logging.debug("node %d receiving." % (self.id))
        self._aggregate(msg_length - cf.HEADER_LENGTH)
        self.amount_received += msg_length

    def _aggregate(self, msg_length):
        logging.debug("node %d aggregating." % (self.id))
        aggregation_cost = self.aggregation_function(msg_length)
        self.tx_queue_size += aggregation_cost        
       
class Node(object):
    def __init__(self, id, parent=None, ip=None):
        self.pos_x = np.random.uniform(0, cf.AREA_WIDTH)
        self.pos_y = np.random.uniform(0, cf.AREA_LENGTH)
        self.ip = ip  # Already supports IP assignment
        self.flow_table = {}
        self.id = id 
       
        if id == cf.BSID:
            self.energy_source = PluggedIn(self)
        else:
            self.energy_source = Battery(self)
        self.id = id
        self.network_handler = parent
        self.timer_logs = OrderedDict()
        self.transitions = []
        self.predicted_transitions = []
        self.energy_map = []
        self.predicted_total_energy_consumed = 0
        self.predicted_remain_energy_list = []
        transition_matrix = None
        self.scheduler = BackgroundScheduler
        self.cluster_head_times = []
        self.node_stage = Stage.SENSING
        my_filter = NoRunningFilter()
        self.log_stage()
        self.reactivate()
        self.markov_energy = 0
        self.Q = 0
        self.MR = 0 #measure of regularity
        self.NB = 0 #number of alive nodes no need s already a a property of the network. 
        self.N_RB = 0 #N_RB is the number of regular neighbors 
        self.DC = 0
        self.trackker_from_markov_interval = 0
        self.w_dist = 0 # stores the distance of a node with other nodes in a cluster
        self.w_energy = 0 # stores the weight of energy of a node
        self.weight = 0 # sum of weights
        self.temp_table = {}
        self.temp_list = []

    def __repr__(self):
        if self.is_controller:
            return "<Controller %s>" % (self.id)
        if self.id == cf.BSID:
            return "<BS>"
        return "<Node %s energy %s>" % (self.id, float(self.energy_source.energy))

    def __str__(self):
        if self.is_controller:
            return "Controller %s at (%s, %s)" % (self.id, self.pos_x, self.pos_y)
        if self.id == cf.BSID:
            return "BS at (%s, %s)" % (self.pos_x, self.pos_y)
        return "Node %s at (%s, %s) with energy %s" % (self.id, self.pos_x, self.pos_y, self.energy_source.energy)

    @property
    def is_controller(self):
        return isinstance(self.energy_source, PluggedIn) and self.id != cf.BSID
    
    def reactivate(self):
        """Reactivate nodes for next simulation."""
        self.alive = 1
        self.tx_queue_size = 0
        self._next_hop = cf.BSID
        self.distance_to_endpoint = 0
        self.amount_sensed = 0
        self.amount_transmitted = 0
        self.amount_received = 0
        self.memership = cf.BSID
        self.aggregation_function = lambda x: 0
        self.time_of_death = cf.INFINITY
        self._is_sleeping = 0
        self.sleep_prob = 0.0
        self.neighbors = []
        self.nb_neighbors = -1
        self.exclusive_radius = 0
        self.flow_table = None
        self.packet_received =0
        self.node_stage = Stage.SENSING
        self.log_stage()


    @property
    def next_hop(self):
        if  self.flow_table and self.id != cf.BSID:
            return self.flow_table[self.id]["destination"]
        return self._next_hop

    @next_hop.setter
    def next_hop(self, value):
        self._next_hop = value
        distance = calculate_distance(self, self.network_handler[value])
        self.distance_to_endpoint = distance

    @property
    def is_sleeping(self):
        if self.is_head():
            self._is_sleeping = 0
        return self._is_sleeping

    @is_sleeping.setter
    def is_sleeping(self, value):
        """Cluster heads cannot be put to sleep."""
        self._is_sleeping = value if not self.is_head() else 0

    def _record_time(Stage=None):
        """ A decorator that measures the time of execution for the method """
        def decorator(func):
            def wrapper(self, *args, **kwargs):
                if not Stage:
                    raise "Stage must be defined to measure time"
                timer = Timer(self.id, Stage)
                self.transitions.append(Stage)
                
                x = func(self, *args, **kwargs)
                timer.stop()
                self.timer_logs[timer.uuid] = timer

            return wrapper
        return decorator

   # @_record_time(Stage=Stage.SLEEPING)
    def set_sleeping_stage(self):
        """ Set node as sleeping for this Stage"""
        self.energy_source.consume(cf.E_SLEEP)
        self.node_stage = Stage.SLEEPING
        
        self.log_stage()
        pass

    #@_record_time(Stage=Stage.IDLE)
    def set_idle_stage(self):
        """ Set node as idle for this Stage"""
        
        self.energy_source.consume(cf.E_IDLE)
        self.node_stage = Stage.IDLE
        self.log_stage()
        self.set_sleeping_stage()
        pass


    def _only_active_nodes(func):
        """This is a decorator. It wraps all energy consuming methods to
        ensure that only active nodes execute this method. Also it automa-
        tically calls the battery.
        """

        def wrapper(self, *args, **kwargs):
            if self.alive and not self.is_sleeping:
                func(self, *args, **kwargs)
                return 1
            else:
                return 0
        return wrapper

    def is_head(self):
        if self.next_hop == cf.SUBCONT0:
            return 1
        elif self.next_hop == cf.SUBCONT1:
            return 1
        else:
            return 0

    def is_ordinary(self):
        return 1 if self.next_hop != cf.BSID and self.id != cf.BSID else 0

    @_only_active_nodes
    #@_record_time(Stage=Stage.AGGREGATING)
    def _aggregate(self, msg_length=None):
        if not msg_length:
            msg_length = cf.MSG_LENGTH
        if self.node_stage == Stage.IDLE:
            self.energy_source.consume(cf.E_IDLE)
        self.node_stage = Stage.AGGREGATING
        self.log_stage()
        logging.debug("node %d aggregating." % (self.id))
        aggregation_cost = self.aggregation_function(msg_length)
        self.tx_queue_size += aggregation_cost
        energy = cf.E_DA * aggregation_cost
        self.energy_source.consume(cf.E_PROCESSING * self.amount_received * msg_length)
       

    @_only_active_nodes
    #@_record_time(Stage=Stage.TRANSMITTING)
    def transmit(self, msg_length=None, destination=None, msg_type = 0):   
        self.node_stage = Stage.TRANSMITTING
        self.log_stage()
        logging.debug("node %d transmitting." % (self.id))
        if not msg_length:
            msg_length = self.tx_queue_size
        # If flow table is defined use destination set on flow table
        if self.flow_table:
            destination = self.flow_table[self.id]["destination"]
            distance = self.distance_to_endpoint
        elif not destination:
            destination = self.network_handler[self.next_hop]
            distance = self.distance_to_endpoint
        else:
            distance = calculate_distance(self, destination)

        # transmitter energy model
        energy = cf.E_ELEC
        if distance > cf.THRESHOLD_DIST:
            energy += cf.E_MP * (distance**4)
        else:
            energy += cf.E_FS * (distance**2)
        energy *= msg_length
        self.amount_transmitted += 1
        self.tx_queue_size = 0
        self.energy_source.consume(cf.E_TRANSMITTING * msg_length)
       

    @_only_active_nodes
    #@_record_time(Stage=Stage.RECEIVING)
    def receive(self, msg_length, msg_type = 0):
        self.node_stage = Stage.RECEIVING
        self.log_stage()
        logging.debug("node %d receiving." % (self.id))  
        self.amount_received += msg_length
        self.packet_received += 1
        energy = cf.E_ELEC * msg_length
        self.energy_source.consume(cf.E_RECEIVING * msg_length)
    
    @_only_active_nodes
    #@_record_time(Stage=Stage.SENSING)
    def sense(self):
        self.node_stage = Stage.SENSING
        self.log_stage()
        logging.debug("node %d sensing." % (self.id))
        if self.node_stage == Stage.AGGREGATING:
            self.energy_source.consume(cf.E_PROCESSING_SENSING)
        self.tx_queue_size = cf.MSG_LENGTH
        self.amount_sensed += cf.MSG_LENGTH
        self.energy_source.consume(cf.SENSING_ENERGY)
        #TEMP
        self.temp = float("{:.2f}".format(np.random.randint(20, 30) + np.random.random()))
        self.temp_list.append(self.temp)
        self.set_idle_stage()
       
    
    def set_waking_stage(self):
        self.node_stage = Stage.SENSING

    def battery_depletion(self):
        self.alive = 0
        self.sleep_prob = 0.0
        self.time_of_death = self.network_handler.round
        self.network_handler.deaths_this_round +=1

    def generate_transition_matrix(self):
        self.transition_matrix = markov_model.generate_transition_matrix(self.transitions)
        self.energy_source.consume(cf.E_PROCESSING)
        self.transmit(destination = self.network_handler[self.next_hop], msg_type = 1)
        return self.transition_matrix

    def log_stage(self):
        # print('logged Stage -> ', self.node_stage)
        if not self.node_stage:
            raise "Stage must be defined to measure time"
        timer = Timer(self.id, self.node_stage)
        self.transitions.append(self.node_stage)
        if timer == cf.TIME_STEP:
            timer.stop()
        self.timer_logs[timer.uuid] = timer
        
