import queue
import threading
import time

class Singleton(type):
    _instance = None

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class SignalManager(metaclass=Singleton):
    sig_map = {}
    asynq = queue.Queue()

    def __init__(self):
        t = threading.Thread(target=self.__listen)
        t.daemon = True
        t.start()

    def __listen(self):
        while True:
            if self.asynq.empty():
                time.sleep(3)
                continue

            signal, args, kwargs = self.asynq.get()
            self.emit(signal, *args, **kwargs)

    def connect(self, signal, slot):
        '''
        Connect signal with slot to receive message

        :param signal:
        :param slot:
        :return:
        '''
        if signal not in self.sig_map.keys():
            self.sig_map[signal] = []
        self.sig_map[signal].append(slot)

    def disconnect(self, signal, slot):
        '''
        Disconnect signal message

        :param signal:
        :param slot:
        :return:
        '''
        if signal in self.sig_map.keys():
            if slot in self.sig_map[signal]:
                self.sig_map[signal].remove(slot)

    def emit(self, signal, *args, **kwargs):
        '''
        Synchronous emission

        :param signal:
        :param args:
        :param kwargs:
        :return:
        '''
        if signal in self.sig_map.keys():
            for s in self.sig_map[signal]:
                try:
                    s(*args, **kwargs)
                except Exception as e:
                    print(e)

    def amit(self, signal, *args, **kwargs):
        '''
        Asyncrhonous emission. Immediately return. No context hang.

        :param signal:
        :param args:
        :param kwargs:
        :return:
        '''
        self.asynq.put([signal, args, kwargs])

    def nmit(self, signal, *args, **kwargs):
        '''
        N thread asynchronus emission. Immediately return. No context hang.

        :param signal:
        :param args:
        :param kwargs:
        :return:
        '''
        t = threading.Thread(target=lambda: self.emit(signal, *args, **kwargs))
        t.daemon = True
        t.start()
        # t.join()


sigmgr = SignalManager()


def slot_no_args():
    print('received!!!!')


def slot_sig1(**kwargs):
    print(kwargs)


def slot_sig2(*args, **kwargs):
    print(args)
    print(kwargs)
    time.sleep(1)


if __name__ == '__main__':
    sigmgr.connect('korea', slot_no_args)
    sigmgr.emit('korea')

    

    # sigmgr.connect('test', slot_sig1)
    # sigmgr.emit('test', data='hello')

    # sigmgr.connect('test2', slot_sig2)

    # # one thread async > time delay 1 sec by each call
    # sigmgr.amit('test2', (1, 2, 3, 4,), data='hello')
    # sigmgr.amit('test2', (1, 2, 3, 4,), data='hello')
    # sigmgr.amit('test2', (1, 2, 3, 4,), data='hello')
    # print('aync emit !')
    # time.sleep(10)

    # # n thread async > output right away all call
    # sigmgr.nmit('test2', (1,2,3,4,), data='hello')
    # sigmgr.nmit('test2', (1, 2, 3, 4,), data='hello')
    # sigmgr.nmit('test2', (1, 2, 3, 4,), data='hello')
    # print('n thread aync emit !')
    # time.sleep(10)
    # exit(0)