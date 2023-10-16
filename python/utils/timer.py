""" Timer implementation to measure time expended at each stage: Transmitting, aggregating, sleeping etc """
import time
import logging
import uuid
import aenum as enum
import config as cf


class Stage(enum.Enum):
    """ Different Stages for the system """
    # The Values are used as indexes for Markov Chain Predictor
    TRANSMITTING = 0
    AGGREGATING = 1
    SLEEPING = 2
    SENSING = 3
    RECEIVING = 4
    IDLE = 5

    @property
    def energy_consumed(self):
        """ Return the energy consumption of the stage """
        if self == Stage.SLEEPING:
            return 0
        if self == Stage.SENSING:
            return cf.SENSING_ENERGY
        if self == Stage.TRANSMITTING:
            return cf.E_FS
        if self == Stage.RECEIVING:
            return cf.E_ELEC
        if self == Stage.AGGREGATING:
            return cf.E_DA
        if self == Stage.IDLE:
            return cf.E_IDLE
        return 0

    def __radd__(self, stage):
        if isinstance(stage, Stage):
            return self.energy_consumed + stage.energy_consumed
        return self.energy_consumed + stage




class Timer:
    def __init__(self, node_id, stage):
        self._start_time = None
        self.node_id = node_id
        self.uuid = str(uuid.uuid4())
        self.elapsed_time = None
        # stage is one of transmitting, aggregating, sleeping.................
        self.stage = stage
        self.t  = time.time()
        self.start()

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError("Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")
        self.elapsed_time = time.perf_counter() - self._start_time
        # print('elapsed time -->',self.elapsed_time)
        self._start_time = None
        logging.debug("Node %s -> Elapsed time %s: %0.4f seconds" %
                      (self.node_id, self.stage, self.elapsed_time))
