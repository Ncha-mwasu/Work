import csv
import logging
import os
import time
import uuid
from datetime import datetime
from multiprocessing.dummy import Pool as ThreadPool

import matplotlib.pyplot as plt
import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
from numpy.core.fromnumeric import trace
from sklearn.metrics import mean_absolute_error

import config as cf
from python.network.node import Controller, Node
from python.network.sleepscheduler import SleepScheduler
#from python.sleep_scheduling.sleep_scheduler import *
from python.utils.grid import *
from python.utils.tracer import *
from python.utils.utils import calculate_distance, get_result_path

"""
Consolidating errors
"""

from collections import OrderedDict

import errors_consolidate.errors_consolidator as ec

"""
Consolidating errors
"""


class Network(list):
    """This class stores a list with all network nodes plus the base sta-
    tion. Its methods ensure the network behavior.
    """
    
    def __init__(self, cont_nodes=None, sensor_nodes =None):
        logging.debug('Instantiating nodes...')
        self.controller_list = cont_nodes
        if sensor_nodes:
           self.extend(sensor_nodes)
        
        if cont_nodes:
           nodes = [Node(i, self) for i in range(0, cf.NB_NODES)]
           self.extend(nodes)
           self.extend(cont_nodes)
           base_station = Node(cf.BSID, self)
           base_station.pos_x = cf.BS_POS_X
           base_station.pos_y = cf.BS_POS_Y
           self.append(base_station)
           
            # last node in nodes is the super controller/base station
            #include sub-controllers and append them to the
        
        self.death_list=[]
        self._dict = {}
        for node in self:
            self._dict[node.id] = node

        self.id = str(uuid.uuid4())
        self.perform_two_level_comm = 1
        self.round = 0
        self.centroids = []
        self.routing_protocol = None
        self.sleep_scheduler_class = None
        self.cnt = 0
        self.initial_energy = self.get_remaining_energy()
        self.first_depletion = 0
        self.per50_depletion = 0
        self.energy_spent = []
        self.packetloss = []
        self.scenario = []
        """
        Consolidating errors
        """
        
        self.mse = []
        self.mae = []
        self.delay=[]
        self.packet_loss =[]
        Network.remaining_energies_path = os.path.join(get_result_path(), 'remaining_energies')
        Network.round_energies_result_csv = {}
        Network.round_energies_result_csv_writers = {}
        x = [node for node in self if isinstance(node, Controller)== False]
        x = [node for node in x if node.alive]
        x = [node for node in x if node.id != cf.BSID]
        Network.all_ordinary_nodes = [node for node in x if node.is_head() == False]
        
        Network.actual_rows = {}
        Network.predicted_rows = {}
        for node in Network.all_ordinary_nodes:
            Network.actual_rows[f'node_{node.id}'] = []
            Network.predicted_rows[f'node_{node.id}'] = []

        """
        Consolidating errors
        """

        self.pack = 0
        self.mse = []
        self.markov_energies = 0
        
        self.sched = BackgroundScheduler()
        self.sched.add_job(self.logger, 'interval', seconds=cf.TIME_STEP)
        self.sched.start()
        self.logger()

    def logger(self):
        for node in self.get_ordinary_nodes():
            node.log_stage()

    def reset(self):
        """Set nodes to initial state so the same placement of nodes can be
        used by different techniques.
        """
        for node in self:
            if isinstance(node, Controller) == False:
                node.energy_source.recharge()
                node.reactivate()

        # allows for updates of BS position between simulations
        self[-1].pos_x = cf.BS_POS_X
        self[-1].pos_y = cf.BS_POS_Y

        self.round = 0
        self.centroids = []
        self.energy_spent = []

        self.routing_protocol = None
        self.sleep_scheduler_class = None
        self.packetloss =[]
        self.death_list=[]
        self.first_depletion = 0
        self.per50_depletion = 0
        self.last_depletion = None
        self.perform_two_level_comm = 1
        self.delay=[]
        self.packet_loss =[]
        # self.deaths_this_round = 0


    @classmethod
    def make_round_energies_result_dir(cls):
        """Mkdir results dir if it doesn't exist"""
        Network.this_remaining_energies_result_dir = os.path.join(get_result_path(), 'remaining_energies')
        dir_path = Network.this_remaining_energies_result_dir
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        # return dir_path

    @classmethod
    def open_round_energies_result_csv(cls, scenario):
        """Write only open the remaining_energies.csv"""
        Network.this_remaining_energies_result_path = os.path.join(Network.this_remaining_energies_result_dir, f'{scenario}_' + datetime.today().strftime('%H-%M-%S') + '_remaining_energies.csv')
        Network.round_energies_result_csv =  open(Network.this_remaining_energies_result_path, mode='w')
        Network.round_energies_result_csv_writer = csv.writer(Network.round_energies_result_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    @classmethod
    def close_round_energies_result_csv(cls):
        """close remaining_energies.csv"""
        Network.round_energies_result_csv.close()


    @classmethod
    def write_round_energies_header_csv(cls):
        """Writes the headings of the remaining_energies.csv"""
        header = [f'node_{i}' for i in range(0, cf.MAX_ROUNDS)]
        Network.round_energies_result_csv_writer.writerow(header)
        # pass

    @classmethod
    def write_round_energies_csv(cls, round_energies):
        """Append the round energies record(single line) to the csv"""
        if not round_energies or not isinstance(round_energies, list) or not isinstance(round_energies[0], str):
            raise Exception('round_energies is a list of strings (all node energies for that round)')
        Network.round_energies_result_csv_writer.writerow(round_energies)


    def simulate(self, scenario):
        tracer = Tracer()

        self.routing_protocol.pre_communication(self)

        all_alive = 1
        percent70_alive = 1
        self.deaths_this_round = 0

        for round_nb in range(0, cf.MAX_ROUNDS):
            if round_nb == 0:
                Network.make_round_energies_result_dir()
                Network.open_round_energies_result_csv(scenario)
                Network.write_round_energies_header_csv()
            elif round_nb >= cf.MAX_ROUNDS:
                Network.close_round_energies_result_csv()

            if self.someone_alive():
                self.round = round_nb
                print_args = (
                        str(self.routing_protocol),
                        round_nb, self.get_remaining_energy())
                print("%s:: round %d: total remaining energy: %f" % print_args)
                nb_alive_nodes = self.count_alive_nodes()
                if nb_alive_nodes == 0:
                    break
                tracer['alive_nodes'][2].append(nb_alive_nodes)
                if cf.TRACE_ENERGY:
                    tracer['energies'][2].append(self.get_remaining_energy())
                    
                self.routing_protocol.setup_phase(self, round_nb)
                
                with open('roundprogress.txt', 'a') as f:
                    f.write(str(len(self.get_alive_nodes()[:-2]))+ '\n')
                # check if someone died
                if self.deaths_this_round != 0:
                    if all_alive == 1:
                        all_alive = 0
                        self.first_depletion = round_nb
                    if float(nb_alive_nodes)/float(cf.NB_NODES) < 0.5 and \
                        percent70_alive == 1:
                        percent70_alive = 0
                        self.per50_depletion = round_nb
                        # Plot the network when 30% has depleted
                        self.routing_protocol.plot_network(round_nb)

                    if not nb_alive_nodes:
                        self.last_depletion = round_nb

                # clears dead counter
                self.deaths_this_round = 0
                #self.routing_protocol.broadcast(self)

                
                self._run_round(round_nb)

        timer_logs = {}
        for node in self:
            if isinstance(node,Node):
                timer_logs.update(node.timer_logs)
                print(node.energy_map)
        try:
            tracer['first_depletion'][2].append(self.first_depletion)
            tracer['50per_depletion'][2].append(self.per50_depletion)
            tracer['last_depletion'][2].append(self.last_depletion)
        except:
            pass

        '''
        Consolidating Errors
        '''

        #this_ten = cf.MARKOV_PREDICTION_INTERVAL
        this_error_mae = np.mean(self.mae)
        this_error_mse = np.mean(self.mse)

       # this_errors_string = f'"{this_ten}p{this_ten}": {{"markov": {{"mae": {this_error_mae}, "mse": {this_error_mse} }}}},\n' if this_ten + 12 < cf.ARIMA_END else f'"{this_ten}p{this_ten}": {{"markov": {{"mae": {this_error_mae}, "mse": {this_error_mse} }}}}\n'

        with open('errors_consolidate_parallel.txt', 'a') as f: 
            f.write(this_error_mse)
        '''
        Consolidating Errors
        '''

        '''
        Code below for aggregate arima results
        '''

        # make the aggregate directory
        Network.arima_aggregate_path = os.path.join(Network.remaining_energies_path, scenario, 'arima')
        if not os.path.exists(Network.arima_aggregate_path):
            os.makedirs(Network.arima_aggregate_path)

        # open the csvs
        Network.this_actual_remaining_energies_result_path = os.path.join(Network.arima_aggregate_path, 'actual_remaining_energies.csv')
        Network.this_predicted_remaining_energies_result_path = os.path.join(Network.arima_aggregate_path, 'predicted_remaining_energies.csv')
        Network.actual_energies_result_csv =  open(Network.this_actual_remaining_energies_result_path, mode='w')
        # Network.predicted_energies_result_csv =  open(Network.this_predicted_remaining_energies_result_path, mode='w')
        
        # make the writers
        Network.actual_round_energies_result_csv_writer = csv.writer(Network.actual_energies_result_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # Network.predicted_round_energies_result_csv_writer = csv.writer(Network.predicted_energies_result_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # write the headers
        header = ['node'] + [f'round_{round_nb}' for round_nb in range(0, cf.MAX_ROUNDS)]
        Network.actual_round_energies_result_csv_writer.writerow(header)
        # Network.predicted_round_energies_result_csv_writer.writerow(header)


        nodes = OrderedDict()
        # Initialize all nodes with an empty list
        for i in range(0, len(Network.all_ordinary_nodes)):
            nodes[i] = None

        # node_ids = []
        i = 0
        for node in Network.all_ordinary_nodes:
            # write the actual_energies rows
            Network.actual_round_energies_result_csv_writer.writerow([node.id] + Network.actual_rows[f'node_{node.id}'])
            # node_ids.append(str(node.id))
            nodes[i] = node
            i = i + 1
        cost1 = np.mean(self.mse) 
        with open('cost.txt', 'a+') as f:
            f.write(str(cost1)+'\n')
        cost2 = np.mean(self.mae)
        with open('cost2.txt', 'a+') as f:
            f.write(str(cost2)+'\n')

        self.sched.shutdown()
        
        return tracer, timer_logs #before optimization
        # return np.mean(self.mse) + np.array([node.markov_energy for node in self.get_ordinary_nodes()]).mean()

    
   # def packet_loss_plot(self):
        
      #  plt.plot(range(0, cf.MAX_ROUNDS), self.packetloss, 'r', label=self.scenario)
      #  plt.xlabel('Number of rounds')
     #   plt.ylabel('Average Packet Loss')
      #  plt.legend(fontsize=11)
      #  plt.show()

        

    def get_packet_loss(self):
        return self.packetloss

    def _run_round(self, round):
        """Run one round. Every node captures using its sensor. Then this
        information is forwarded through the intermediary nodes to the base
        station.
        """
        before_energy = self.get_remaining_energy()
        self.current_time = time.time()
        #while time.time()-self.current_time <= 50*cf.TIME_SLOT:
        for i in range(0, cf.MAX_TX_PER_ROUND):
            self._sensing_phase()
            t0 = time.time()
            self._communication_phase()
            self.delay.append((time.time()-t0))
                    # total_transmitted_per_round_to_CH = {}
            total_transmitted_per_round_to_CH = {}
            total_received_per_round_per_CH ={}

            for node in self.get_ordinary_nodes():
        # print(node.next_hop)
                total_transmitted_per_round_to_CH['cluster_head_id '+str(node.next_hop)] = 0
                    
            for node in self.get_ordinary_nodes():
                # node.transmit()
                total_transmitted_per_round_to_CH['cluster_head_id '+str(node.next_hop)] += node.amount_transmitted
    
            with open('transmitted.txt', 'a') as f:
                    f.write(str(total_transmitted_per_round_to_CH)+'\n')

            # for node in self.get_ordinary_nodes():    
            #     for nde in self:
            #         if nde.id == node.network_handler[node.next_hop].id:
            #         #  print('------------')
            #         # print(nde.amount_received)
            #             total_received_per_round_per_CH ['cluster_head_id '+str(node.next_hop)] = nde.packet_received
            #             #print(total_received_per_round_per_CH)
            #             nde.transmit()
            for node in self.get_heads():
                total_received_per_round_per_CH ['cluster_head_id '+str(node.id)] = 0
                
            for node in self.get_heads():
                total_received_per_round_per_CH ['cluster_head_id '+str(node.id)] += node.packet_received
                node.transmit()

                #node._aggregate()
            #print(total_received_per_round_per_CH)       
            packet_loss_per_round_cluster = {}
            self.packet_loss_total = 0
            self.pack = 0
            packetloss_head = []
            self.number_of_CH = len(total_received_per_round_per_CH.keys())
            for head in total_received_per_round_per_CH.keys():
                for node in total_transmitted_per_round_to_CH.keys():
                    if head == node:
                        packet_loss =  total_transmitted_per_round_to_CH[head] - total_received_per_round_per_CH[head] 
                        # print(total_transmitted_per_round_to_CH[head],total_received_per_round_per_CH[head])
                        packet_loss_per_round_cluster[head] = packet_loss
                        # with open('packetlosses.txt', 'a') as f:
                        #     f.write('average packet loss for round '+str(round)+' '+ ' packet_loss_total '+str(packet_loss)'\n')

                        self.packet_loss_total += packet_loss
                        packetloss_head.append(packet_loss)
            average_heads = np.mean(packetloss_head, axis=0)
            self.packetloss.append(average_heads)
             # populate actual and predicted rows for use in csv files
            for node in Network.all_ordinary_nodes:
                round_energy = str(node.energy_source.energy)
                Network.actual_rows[f'node_{node.id}'].append(round_energy)
    
        # write record (single line) into remaining_energies.csv
        round_energies = [str(node.energy_source.energy) for node in self.get_ordinary_nodes()]
        Network.write_round_energies_csv(round_energies)
        y_truth = []
        y_predicted = []
        if  round ==0:
            with open('MeanSquaredErrors.txt', 'a') as f:
                    f.truncate(0)

            
            with open('MeanAbsoluteErrors.txt', 'w') as f:
                    f.truncate(0)

        '''
        Consolidating Errors
        '''        

        '''if round >= cf.MARKOV_PREDICTION_INTERVAL:
            mse_errors_list = []
            mae_errors_list = []
            for node in self.get_ordinary_nodes():
                forecast = node.predicted_remain_energy_list[round%cf.MARKOV_PREDICTION_INTERVAL]
                actual = node.energy_source.energy
                mse_errors_list.append(pow(forecast - actual, 2))
                mae_errors_list.append(np.abs(forecast - actual))
                y_truth.append(node.energy_source.energy)
                y_predicted.append(node.predicted_remain_energy_list[round%cf.MARKOV_PREDICTION_INTERVAL])
            with open('MeanSquaredErrors.txt', 'a') as f:
                f.write(str(np.mean(mse_errors_list))+'\n')
                self.mse.append(np.mean(mse_errors_list))
            with open('MeanAbsoluteErrors.txt', 'a') as f:
                f.write(str(mean_absolute_error(y_predicted, y_truth))+'\n')
                self.mae.append(np.mean(mae_errors_list))'''

        '''
        Consolidating Errors
        '''
        
        for node in self.get_ordinary_nodes():
            self.markov_energies += node.markov_energy
            
        self.pack = self.packet_loss_total
        self.death_list.append(self.deaths_this_round)
        # For nodes that are asleep, set the current stage as sleeping for tracking
        for node in self.get_sleeping_nodes():
            node.set_sleeping_stage()

        after_energy = self.get_remaining_energy()
        self.energy_spent.append(before_energy - after_energy)

        if round == cf.MAX_ROUNDS-1:
            self.pack =0 

        
        

    def _sensing_phase(self):
        """Every alive node captures information using its sensor."""
        for node in self.get_active_nodes():
                if node != cf.BSID:
                    if isinstance(node,Controller)==False:
                        node.sense()

    def _communication_phase(self):
        """Each node transmits respecting its hierarchy: leaves start the
        communication, then cluster heads forward the messages, until all
        messages reach the base station. This method works for any hierar-
        chy (even for LEACH).
        """
        #ordinary_nodes = self.get_ordinary_nodes()
        #heads = self.get_ch_nodes()
        #msg = str("%d ordinary nodes, %d heads." % (len(ordinary_nodes), len(heads)))
        #logging.debug("Hierarchical communication: %s" % (msg))

        alive_nodes = self.get_alive_nodes()
       # print('next hop of alive nodes')
        for i in alive_nodes:
            if isinstance(i, Node):
                print(i.next_hop)
        # print('alive nodes', alive_nodes)
        if self.perform_two_level_comm == 1:
            self._two_level_comm(alive_nodes)
        else:
            self._recursive_comm(alive_nodes)

    def _recursive_comm(self, alive_nodes):
        """Hierarchical communication using recursivity. This method suppo-
        ses that there is no cycle in the network (network is a tree).
        Otherwise, expect infinite loop.
        """
        next_alive_nodes = alive_nodes[:]
        for node in alive_nodes:
            # check if other nodes must send info to this node
            depends_on_other_node = 0
            for other_node in alive_nodes:
                # if other_node == node:
                #  continue
                if other_node.next_hop == node.id:
                    depends_on_other_node = 1
                    break

            if not depends_on_other_node:
                node.transmit()
                next_alive_nodes = [n for n in next_alive_nodes if n != node]

        if len(next_alive_nodes) == 0:
            return
        else:
            self._recursive_comm(next_alive_nodes)
    cnt = 0
    
    def _two_level_comm(self, alive_nodes):
        
        """This method performs communication supposing that there are only
        ordinary nodes and cluster heads, this method is less generic than
        its recursive version, but it is faster.
        """
        # heads wait for all ordinary nodes, then transmit to BS 
        x = [node for node in alive_nodes if isinstance(node, Controller)== False]
        x = [node for node in x if node.alive]
        x = [node for node in x if node.next_hop != -3]
        x = [node for node in x if node.next_hop != -4]
        x = [node for node in x if node.id != cf.BSID]
        #print(x)
       
        # print(x)
       # ordinary_nodes =  [node for node in x if  node.is_ordinary()]
        ordinary_nodes = x

        # total_transmitted_per_round_to_CH = {};
        # total_received_per_round_per_CH ={};
        #ordinary_nodes = self.get_ordinary_nodes()
        #print('len of ord nodes ', len(ordinary_nodes))

 
        # scheduler = SleepScheduler(cluster_nodes= ordinary_nodes)

        cluster_ids = set([x.next_hop for x in ordinary_nodes])
       # print('clusters', cluster_ids)

        all_clusters = []
        for i in cluster_ids:
            cluster_nodes = []
            for node in ordinary_nodes:
                if node.next_hop == i:
                    cluster_nodes.append(node)
            all_clusters.append(cluster_nodes)
        #scheduler = SleepScheduler(cluster_nodes = cluster_nodes)
        #print('clusters', str(all_clusters))
        # for node in ordinary_nodes:
        #DC_ALGORITHM Proposed
        # for cluster in all_clusters:
        #     total_available_nodes_in_cluster = 0
        #     bad_nodes_in_cluster = []
        #     for node in cluster:
        #         if node.trackker_from_markov_interval <= round(node.DC * cf.MARKOV_PREDICTION_INTERVAL): # check if node has exceeded its duty cycle for markov interval rounds
        #             total_available_nodes_in_cluster += 1
        #         else:
        #             node.set_sleeping_stage()
        #             bad_nodes_in_cluster.append(node)
        #     if total_available_nodes_in_cluster >= round(0.5  * len(cluster)): # confirm if 50% of cluster is available
        #         [node.set_sleeping_stage() for node in bad_nodes_in_cluster] #make those nodes to rest
        #         [cluster.remove(node) for node in bad_nodes_in_cluster]
        #     else:
        #         while total_available_nodes_in_cluster < round(0.5 * len(cluster)):
        #             bad_nodes_in_cluster.remove(bad_nodes_in_cluster[0])
        #             total_available_nodes_in_cluster += 1
        #         [node.set_sleeping_stage() for node in bad_nodes_in_cluster] #make those nodes to rest
        #         [cluster.remove(node) for node in bad_nodes_in_cluster]
                
        # sleepers = []
        # for cluster in all_clusters:
        #     cluster_copy = list(cluster)
        #     for node in cluster_copy:
        #         if node.energy_source.energy < (0.75*cf.INITIAL_ENERGY):
        #             destination = node.network_handler[node.next_hop]
        #             node.transmit(msg_length = 4, destination=destination)
        #             node.receive(4)
                    
        #             if len(cluster) > 1:
        #                 if len(cluster) >= round(0.5  * len(cluster)): # confirm if 50% of cluster is available
        #                     node.set_sleeping_stage() # make those nodes to rest
        #                     sleepers.append(node)
        #                     #print('sleepers', str(sleepers))
        #                     cluster.remove(node)
        #                 else:
        #                     node.set_sleeping_stage()
        #                     cluster.remove(node)
        #                     cluster.append(sleepers.pop(0))  # add the first node that slept to clusters
        #                     sleepers.append(node)
        #             else:
        #                 if len(sleepers) > 0:
        #                     cluster.append(sleepers.pop(0))  # add the first node that slept to clusters
        
        '''sleepers = []
        n_alive = 0
        l_energy = []
        l_dist = []
        w = 0
        for cluster in all_clusters:
            cluster_copy = list(cluster)
            for node in cluster_copy:
                destination = node.network_handler[node.next_hop]
                node.transmit(msg_length=5, destination=destination)
                node.receive(msg_length=5)
                
                if node.energy_source.energy < (0.75*cf.INITIAL_ENERGY):
                    node.set_sleeping_stage()
                    cluster.remove(node)
                    sleepers.append(node)
                    n_alive -= 1
                else:
                    node.w_dist = sum([calculate_distance(node, next_node)**2 for next_node in cluster])
                    n_alive += 1
            
            l_energy.__add__(cluster)
                    
        k = (n_alive - len(sleepers)) * 0.05
        # sort energy level of alive nodes in ascending order
        l_energy.sort(key=lambda node: node.energy_source.energy)
        # sort energy level of alive nodes in ascending order
        l_dist = list(l_energy)
        l_dist.sort(key=lambda node: node.w_dist, reverse=True)
        
        for i in range(0, len(l_energy)):
            # give nodes sequential weights from 1 up to the highest node
            l_energy[i].w_energy = i+1
            l_dist[i].w_dist = i+1 
        w = sum([node.energy_source.energy + node.w_dist for node in l_dist])'''
        
        for cluster in all_clusters:
            # def num_of_awake_nodes():
            #         return sum([1 for node in cluster if not node.is_sleeping])
            
            for node in cluster:
                destination = node.network_handler[node.next_hop]
                # node.transmit(msg_length=5, destination=destination)
                # node.receive(msg_length=5)
                if node.temp_list[-1] < cf.DCCHP_THRESH:
                    node.is_sleeping_stage=True
                    node.set_sleeping_stage()
                if node.is_head:
                     node.is_sleeping = False
                     node.set_waking_stage()
                    
                # if node.is_sleeping:
                #     destination = node.network_handler[node.next_hop]
                #     node.transmit(msg_length=5, destination=destination)
                #     node.receive(msg_length=5)
                
                # if node.energy_source.energy < (0.75*cf.INITIAL_ENERGY):
                #     node.is_sleeping_stage=True
                #     node.set_sleeping_stage()
                # elif node.node_stage == node.Stage.SLEEPING and num_of_awake_nodes < cf.MIN_AWAKE_NODE:
                #     node.is_sleeping = False
                #     node.set_waking_stage()
                
        #try: 
            #ordinary_nodes[0].transmit()
        #except:
           # pass

        t0 = time.time()
        for cluster in all_clusters:
            scheduler = SleepScheduler(cluster_nodes = cluster)
            #print('nodeID', cluster)
            for node in cluster:
                    #self._sensing_phase()
                    nodez = scheduler.get_next_to_transmit()
                    if node == nodez:
                        destination = node.network_handler[node.next_hop]
                        distance = node.distance_to_endpoint
                        
                        if(time.time()-t0 <= cf.TIME_SLOT):
                            node.transmit(destination=destination)
                            if distance<=cf.DISTANCE_THRESH:
                                destination.receive(cf.MSG_LENGTH) 
                        else:
                            break
                        node.set_idle_stage()
        
        '''for cluster in all_clusters:
            for node in cluster:
                    key_node = scheduler.get_next_to_transmit()                
                    while(time.time()-self.current_time <= cf.TIME_SLOT): 
                        key_node.transmit()
                    key_node.set_idle_stage()'''

        # for node in ordinary_nodes:
        #     total_transmitted_per_round_to_CH['cluster_head_id '+str(node.next_hop)] += node.amount_transmitted
   
        # with open('transmitted.txt', 'a') as f:
        #         f.write(str(total_transmitted_per_round_to_CH)+'\n')

        # for node in self.get_ordinary_nodes():    
        #     for nde in self:
        #         if nde.id == node.network_handler[node.next_hop].id:
        #             total_received_per_round_per_CH ['cluster_head_id '+str(node.next_hop)] = nde.packet_received
        #             nde.transmit()
            
        # with open('received.txt', 'a') as f:
        #         f.write(str(total_received_per_round_per_CH)+'\n')        
        # packet_loss_per_round_cluster = {};
        # packet_loss_total = 0

        # number_of_CH = len(total_received_per_round_per_CH.keys())
        # for head in total_received_per_round_per_CH.keys():
        #     for node in total_transmitted_per_round_to_CH.keys():
        #         if head == node:
        #            packet_loss =  total_transmitted_per_round_to_CH[head] - total_received_per_round_per_CH[head] 
        #            #print('cluster ' +str(head)+' packet loss '+str(packet_loss))
        #            packet_loss_per_round_cluster[head] = packet_loss
        #            packet_loss_total +=packet_loss
        # if self.cnt == 0:
        #    with open('packetloss.txt', 'a') as f:
        #         f.write('average packet loss for round '+str(self.cnt)+' '+str(packet_loss_total/number_of_CH)+'\n')
        #         self.packetloss.append(packet_loss_total/number_of_CH)
        # else:
        #     with open('packetloss.txt', 'a') as f:
        #         f.write('average packet loss for round '+str(self.cnt)+' '+str((packet_loss_total-self.former_total_packet)/number_of_CH)+'\n')
        #         self.packetloss.append((packet_loss_total-self.former_total_packet)/number_of_CH)
        
        
        # #with open('packetloss.txt', 'a') as f:
        #  #       f.write(str(packet_loss_per_round_cluster)+'\n')
        
        # self.former_total_packet = packet_loss
        # self.cnt+=1

    
    '''def calculate_distance(node1, node2):
        """Calculate the Euclidean distance between two nodes."""
        x1 = node1.pos_x
        y1 = node1.pos_y
        x2 = node2.pos_x
        y2 = node2.pos_y
        return calculate_distance_point(x1, y1, x2, y2)'''
    
    def get_alive_nodes(self):
        """Return nodes that have positive remaining energy."""
        #re-write to reflect sub-controllers.
        return [node for node in self[0:-1] if node.alive]

    def get_sleeping_nodes(self):
        """Return nodes that are sleeping"""
        return [node for node in self[0:-1] if node.is_sleeping]

    def set_scenario(self, scenario):
        self.scenario = scenario

    def get_active_nodes(self):
        """Return nodes that have positive remaining energy and that are
        awake."""
        def is_active(
            x): return  x.alive and not x.is_sleeping
        return [node for node in self[0:-1] if is_active(node)]

    def get_ordinary_nodes(self):
        x = [node for node in self if isinstance(node, Controller)== False]
        x = [node for node in x if node.alive]
        x = [node for node in x if node.id != cf.BSID]
        return [node for node in x if node.is_head()== False]



    def get_heads(self, only_alives=1):
        input_set = self.get_alive_nodes() if only_alives else self
        #return [node for node in input_set if node.is_head()]
        input_set = [node for node in input_set if isinstance(node,Node)]
        return [node for node in input_set if node.is_head()]

    def get_sensor_nodes(self):
        """Return all nodes except base station."""
        return [node for node in self[0:-1]]

    def get_average_energy(self):
        return np.average(self.energy_spent)

    def someone_alive(self):
        """Finds if there is at least one node alive. It excludes the base station,
           which is supposed to be always alive."""
        for node in self[0:-3]:
            if node.alive == 1:
                return 1
        return 0

    def count_alive_nodes(self):
        return sum(x.alive for x in self[:-3])

    def get_BS(self):
        # intention: make code clearer for non-Python readers
        return self[-1]

    def get_node(self, id):
        """By default, we assume that the id is equal to the node's posi-
        tion in the list, but that may not be always the case.
        """
        return self._dict[id]

    def notify_position(self):
        """Every node transmit its position directly to the base station."""
        for node in self.get_alive_nodes():
            node.transmit(msg_length=cf.MSG_LENGTH, destination=self.get_BS())

    def broadcast_next_hop(self):
        """Base station informs nodes about their next hop."""
        base_station = self.get_BS()
        for node in self.get_alive_nodes():
            base_station.transmit(msg_length=cf.MSG_LENGTH, destination=node)

    def get_nodes_by_membership(self, membership, only_alives=1):
        """Returns all nodes that belong to this membership/cluster."""
        input_set = self.get_alive_nodes() if only_alives else self
        def condition(
            node): return node.membership == membership and node.id != cf.BSID
        return [node for node in input_set if condition(node)]

    def get_remaining_energy(self, ignore_nodes=None, network=None):
        """Returns the sum of the remaining energies at all nodes."""
        if not network:
            network = self
        set = network.get_alive_nodes()
        if len(set) == 0:
            return 0
        if ignore_nodes:
            set = [node for node in set if node not in ignore_nodes]

        def transform(x): return x.energy_source.energy
        energies = [transform(x) for x in set]
        return sum(x for x in energies)

    def set_aggregation_function(self, function):
        """Sets the function that determines the cost of aggregation."""
        for node in self:
            node.aggregation_function = function

    def split_in_clusters(self, nb_clusters=cf.NB_CLUSTERS):
        """Split this nodes object into other nodes objects that contain only
        information about a single cluster."""
        clusters = []
        for cluster_idx in range(0, nb_clusters):
            nodes = self.get_nodes_by_membership(cluster_idx)
            cluster = Network(init_nodes=nodes)
            cluster.append(self.get_BS())
            clusters.append(cluster)
        return clusters

    def _calculate_nb_neighbors(self, target_node):
        """Calculate the number of neighbors given the sensor coverage
        radius.
        """
        # if number of neighbors was calculated at least once
        # skips calculating the distance
        if target_node.nb_neighbors != -1:
            # only check if there are dead nodes
            all_neighbors = target_node.neighbors 
            nb_dead_neighbors = sum(1 for x in all_neighbors if not x.alive)
            target_node.neighbors[:] = [x for x in all_neighbors if x.alive]
            return target_node.nb_neighbors - nb_dead_neighbors

        nb_neighbors = 0
        shortest_distance = cf.COVERAGE_RADIUS*2
        for node in self.get_alive_nodes():
            if node == target_node:
                continue
            distance = calculate_distance(target_node, node)
            if distance <= cf.COVERAGE_RADIUS:
                nb_neighbors += 1
                target_node.neighbors.append(node)
                if distance < shortest_distance:
                    shortest_distance = distance

        if shortest_distance != cf.INFINITY:
            exclusive_radius = shortest_distance - cf.COVERAGE_RADIUS
            if exclusive_radius < 0:
                exclusive_radius = 0.0

        node.nb_neighbors = nb_neighbors
        node.exclusive_radius = exclusive_radius

    def update_neighbors(self):
        for node in self.get_alive_nodes():
            self._calculate_nb_neighbors(node)

        self.update_sleep_prob()

    def update_sleep_prob(self):
        for node in self.get_alive_nodes():
            node.update_sleep_prob()
   
    def get_controllers(self):
        return self.controller_list