from lib.train.utils import TensorDict

class BaseActor:
    '''
    Base class for actor. The actor class handles the passing of the data through the network and loss calculation
    Actor classes shoud inherit from this one!!
    '''
    def __init__(self, net, objective) -> None:
        '''
        args:
            net - The network to train
            objective - The loss function
        '''
        self.net = net
        self.objective = objective

    def __call__(self, data):
        '''
        args:
            data - A TensorDict containing all the necessary data

        returns:
            loss - loss for the input data
            stats - a dict containing detailed losses
        '''
        raise NotImplementedError()

    def to(self, device):
        self.net.to(device)

    def train(self, mode=True):
        self.net.train(mode)

    def eval(self):
        self.train(False)
        