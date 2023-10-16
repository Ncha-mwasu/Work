
import argparse
import inspect
import logging
import os
import sys
import uuid
from collections import OrderedDict

import matplotlib

# For CI-Environments run matplotlib without any display backend
if os.environ.get("DRY_RUN"):
    matplotlib.use('Agg')

import math

import numpy as np

import config as cf
from python.network.aggregation_model import *
from python.network.network import Network
from python.network.node import Controller
from python.routing.direct_communication import *
from python.routing.fcm import *
from python.routing.leach import *
from python.routing.mleach import *
from python.routing.mte import *
from python.utils.tracer import *
from python.utils.utils import *

logging.basicConfig(stream=sys.stderr, level=logging.INFO)


def run_scenarios(kwargs):
    subcontr0 = Controller(cf.SUBCONT0, cf.SUB_CON0_POS_X, cf.SUB_CON0_POS_Y)
    subcontr1 = Controller(cf.SUBCONT1, cf.SUB_CON1_POS_X, cf.SUB_CON1_POS_Y)
    

    network = Network(cont_nodes=[subcontr0, subcontr1])

    traces = {}
    timer_logs = OrderedDict()
    remaining_energies = []
    average_energies = []
    scenario_names = {}

    scenarios = cf.scenarios if not kwargs.get("dry_run") else cf.dry_run_scenarios

    cf.HETEROGEANOUS = kwargs.get("heterogeanous")

    cf.RESULTS_PATH = os.path.join(cf.RESULTS_PATH, time.strftime("%Y-%m-%d"), time.strftime("%H-%M"))

    # If node initial energy is specified, network is homogeanous
    if kwargs.get("initial_energy"):
        cf.INITIAL_ENERGY = kwargs.get("initial_energy")
        cf.HETEROGEANOUS = False
    list_packet_losses = []
    nicknames_list = []
    total_death=[]

    # nicknames_list = []
    for scenario in scenarios:
        if type(scenario) is str:
            exec(scenario)


            
            continue
        network.set_scenario(scenario[0])
        nicknames_list.append(scenario[0])
       # print(network.packetloss)
        
        network.reset()
        routing_topology, optimization, aggregation, nickname = scenario
        
        if nickname:
            scenario_name = nickname
        else:
            if optimization:
                scenario_name = routing_topology+' + '+optimization
            else:
                scenario_name = routing_topology

        if scenario_name in scenario_names:
            scenario_names[scenario_name] += 1
            scenario_name += " (" + str(scenario_names[scenario_name]) + ")"
        else:
            scenario_names[scenario_name] = 1

        routing_protocol_class = eval(routing_topology)
        network.routing_protocol = routing_protocol_class()
        
        if optimization:
            sleep_scheduler_class = eval(optimization)
            not_class_msg = 'optimization does not hold the name of a class'
            assert inspect.isclass(sleep_scheduler_class), not_class_msg
            network.sleep_scheduler_class = sleep_scheduler_class

            aggregation_function = aggregation + '_cost_aggregation'
            network.set_aggregation_function(eval(aggregation_function))

        logging.info(scenario_name + ': running scenario...')
        try:
            traces[scenario_name], timer_logs[scenario_name] = network.simulate(scenario[0])

            print('what type am i dealing with', type(traces[scenario_name]['energies']))
            print(traces[scenario_name]['energies'])
            remaining_energies.append(600.0 - network.get_remaining_energy())

        except:
            average_energies.append(network.get_average_energy())
            list_packet_losses.append(network.packetloss)
            total_death.append(network.death_list)
            print("Done Simulating")
            break

    
    # print(list_packet_losses, 'Packet loss ===================================')
    
    # colors = ['b-', 'r-', 'k-', 'y-', 'g-', 'c-', 'm-']
    # x = matplotlib.pyplot
    # fig = x.figure()
    # for i, packetloss in enumerate(list_packet_losses):
    #     # x.plot(range(0, len(packetloss)), packetloss, colors[i], label=nicknames_list[i])
    #     # x.xlabel('Number of rounds')
    #     # x.ylabel('Average Packet Loss')
    #     # x.legend(fontsize=11)
    #     averages = np.array_split(packetloss, ceil((len(packetloss)/10))).mean()
    #     ax = fig.add_axes([0,0,1,1])
    #     rounds = [ str(10 + x * 10) for x in range(len(averages))]
    #     ax.bar(rounds,averages, color=colors[i])
    #     x.show()
    # x.show()

    colors = ['b', 'r', 'k', 'y', 'g', 'c', 'm']
    # list_packet_losses = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]

    #new plot
    # a = matplotlib.pyplot
    # Total_average = []
    # huns_split = []
    # for i, packetloss in enumerate(list_packet_losses):
    #     split = np.array_split(packetloss, math.ceil(len(packetloss)/100))
    #     huns_split.append(split)
    #     for hun in enumerate(huns_split):
    #         averages = np.mean(hun)
    #         Total_average.append(averages)
    #         with open('averageloss100.txt', 'a') as f:
    #                 f.write(str(Total_average) +'\n')
    #         #rounds = np.arange(len(averages))
    #         width = 0.8
    #         a.ylabel('Average packet loss')
    #         a.xlabel('Rounds')
    #         #Pos = np.array(range(len(rounds)))
    #         x = ['100', '200', '300', '400']
    #         x_pos = [i for i, _ in enumerate(x)]
    #         a.bar(x_pos, averages, width = width, label=nicknames_list[i])
    #         a.xticks(x_pos, x)
    #     a.legend()
    # a.show()
    means1 = []
    mean1 = 0
    splitedSize = 10
    b_splited = [network.packetloss[x:x+splitedSize] for x in range(0, len(network.packetloss), splitedSize)]
    for i in b_splited:
            mean1 = np.mean(i)
            means1.append(mean1)
    with open('average_packetloss2.txt', 'a+') as f:
                f.write(str(means1) +'\n')
    
    #sum of dead nodes per 100 rounds
    sums = []
    sum = 0
    splitedSize = 10
    a_splited = [network.death_list[x:x+splitedSize] for x in range(0, len(network.death_list), splitedSize)]
    #print('*****', len(a_splited))
    for i in a_splited:
            sum = np.mean(i)
            sums.append(sum)
    with open('average_death_list.txt', 'a+') as f:
                f.write(str(sums) +'\n')

    #energy spent per 1000 rounds
    sums2 = []
    sum2 = 0
    splitedSize = 10
    a_splited = [network.energy_spent[x:x+splitedSize] for x in range(0, len(network.energy_spent), splitedSize)]
    #print('*****', len(a_splited))
    for i in a_splited:
            sum2 = np.mean(i)
            sums2.append(sum2)
    with open('average_energy spent.txt', 'a+') as f:
                f.write(str(sums2) +'\n')

    delays = []
    delay = 0
    splitedSize = 10
    a_splited = [network.delay[x:x+splitedSize] for x in range(0, len(network.delay), splitedSize)]
    #print('*****', len(a_splited))
    for i in a_splited:
            delay = np.mean(i)
            delays.append(delay)
    with open('average_delay.txt', 'a+') as f:
                f.write(str(delays) +'\n')


    if cf.TRACE_COVERAGE:
        print_coverage_info(traces)

    print('Remaining energies: ')
    print(remaining_energies)
    print('Average energies: ')
    print(average_energies)
    return remaining_energies, average_energies


def run_parameter_sweep():

    totals = {}
    avgs = {}

    for network_width in [400, 360, 320, 280, 240, 200, 160, 120, 80, 40]:
        totals[network_width] = {}
        avgs[network_width] = {}
        for elec_energy in [100e-9, 80e-9, 60e-9, 40e-9, 20e-9]:
            cf.AREA_WIDTH = network_width
            cf.AREA_LENGTH = network_width
            cf.BS_POS_X = network_width/2
            cf.BS_POS_Y = network_width/2
            cf.E_ELEC = elec_energy

            remaining_energies, average_energies = run_scenarios()
            totals[network_width][elec_energy] = remaining_energies
            avgs[network_width][elec_energy] = average_energies

    print(totals)
    print(avgs)


if __name__ == '__main__':
    # run_parameter_sweep()
    parser = argparse.ArgumentParser(description='Simulation Environment')
    parser.add_argument('--initial-energy', type=float, help='The Initial Energy of nodes')
    parser.add_argument('-d','--dry-run', action="store_true", help='Run scenarios in Dry Run')
    parser.add_argument('--heterogeanous', action="store_true", help='Nodes should be heterogeanous')
    args = vars(parser.parse_args())
    run_scenarios(args)
