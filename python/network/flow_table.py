import uuid

import config as cf

from .node import Node


class FlowTable(dict):
    """ 
    The flowtable stores configuration of node-node data transfer
    """

    def __init__(self, id, network):
        # the `id` correspond to the cluster_id
        self.id = id
        self.head = network.get_head(id)   # Get Cluster Head
        self.controller = network.get_controller(id)
        self.update_table(id, network)

    def update_table(self, id, network):
        """ Set flow table values """
        for node in network.get_nodes_by_membership(id):
            # Update Flow Table to Current Flow Table
            node.flow_table = self
            # Set all nodes destination in the current cluster to the cluster head
            if node.is_ordinary:
                self[node.id] = {"destination": self.head}
            # Cluster Head destination is set to Controller
            if node.is_head:
                self[node.id] = {"destination": self.controller}

    def rules(self, source_node, packet):
        """ 
        A definition of all rules for the action to be taken by the nodes when data is received """
        pass
