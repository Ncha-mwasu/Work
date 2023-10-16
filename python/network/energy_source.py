import logging

import numpy as np

import config as cf


class EnergySource(object):
    def __init__(self, parent):
        if cf.HETEROGEANOUS:
            # For Heterogeanous Mode, All NOdes have different battery levels
            self.energy = (1 + np.random.rand()) * 0.5
        else:
            self.energy = cf.INITIAL_ENERGY
        self.node = parent

    def recharge(self):
        if not cf.HETEROGEANOUS:
            self.energy = cf.INITIAL_ENERGY
        else:
            self.energy = (1 + np.random.rand()) * 0.5


class Battery(EnergySource):
    def consume(self, energy):
        if self.energy >= energy:
            self.energy -= energy
        else:
            logging.info("node %d: battery is depleted." % (self.node.id))
            self.energy = 0

            self.node.battery_depletion()


class PluggedIn(EnergySource):
    def consume(self, energy):
        pass
