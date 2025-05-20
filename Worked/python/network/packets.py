import numpy as np

import config as cf
from python.network.energy_source import PluggedIn

from .node import Node


class Packet(object):
    """ A packet is a unit of transmission of data
    """

    def __init__(self, id, source_node_id):
        self.id = id
        self.source_node = source_node_id
