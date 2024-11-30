# This file reads config.ini and returns a Configuration object
# The Configuration object contains all the parameters needed for the training

import configparser
import torch.nn as nn

class Configuration:
    def __init__(self, file_path='config.ini'):
        self.config = configparser.ConfigParser()
        self.config.read(file_path)

        self.lr = self.config.getfloat("DEFAULT","lr")
        self.epochs = self.config.getint("DEFAULT","epochs")
        self.batchsize = self.config.getint("DEFAULT","batch_size")
        self.input_dim = self.config.getint("DEFAULT","input_dim")
        self.output_dim = self.config.getint("DEFAULT","output_dim")
        self.log_interval = self.config.getint("DEFAULT","log_interval")

        # Parse hidden layers as an array
        hidden_layers_str = self.config.get("DEFAULT", "hidden_layers")
        self.hidden_layers = [int(x) for x in hidden_layers_str.strip('[]').split(',')]


    def to_dict(self):
        return {
            "lr": self.lr,
            "epochs": self.epochs,
            "batchsize": self.batchsize,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "log_interval": self.log_interval,
            "hidden_layers": self.hidden_layers,
        }
 


        loss_function_name = self.config.get('Loss', 'loss_function')

        # Check if the loss function name is valid and available
        if hasattr(losses, loss_function_name):
            # Get the loss function class dynamically using getattr
            loss_function_class = getattr(losses, loss_function_name)

            # Get parameters from the config
            parameters = {}
            for key in self.config.options('Loss'):
                if key != 'loss_function':
                    parameters[key] = float(self.config.get('Loss', key))

            # Instantiate the loss function with parameters
            return loss_function_class(**parameters)
        else:
            raise ValueError(f"Unsupported loss function: {loss_function_name}")