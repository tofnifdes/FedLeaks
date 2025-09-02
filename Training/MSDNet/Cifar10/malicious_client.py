"""Flower client example using PyTorch for CIFAR-10 image classification."""

import argparse
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader

import cifar
from Msdnet import MSDNet
import flwr as fl

#fl.common.logger.configure(identifier="myFlowerExperiment", filename="log.txt")


disable_progress_bar()


USE_FEDBN: bool = False

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Flower Client
class CifarClient(fl.client.NumPyClient):
    """Flower client implementing CIFAR-10 image classification using PyTorch."""

    def __init__(
        self,
        model: cifar.Net,
        target_model: cifar.Net,
        trainloader: DataLoader,
        testloader: DataLoader,
        client_id:int,
    ) -> None:
        self.model = model
        self.target_model = target_model
        self.trainloader = trainloader
        self.testloader = testloader
        self.client_id = client_id

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        self.model.train()
        if USE_FEDBN:
            # Return model parameters as a list of NumPy ndarrays, excluding
            # parameters of BN layers when using FedBN
            return [
                val.cpu().numpy()
                for name, val in self.model.state_dict().items()
                if "bn" not in name
            ]
        else:
            # Return model parameters as a list of NumPy ndarrays
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        self.model.train()
        if USE_FEDBN:
            keys = [k for k in self.model.state_dict().keys() if "bn" not in k]
            params_dict = zip(keys, parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=False)
        else:
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
            self.target_model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        print("CONFIG",config)
        self.set_parameters(parameters)
        cifar.train_normal(self.model, self.trainloader, epochs=1, device=DEVICE)
        
        if config['start'] == True:

            attribute = config['attribute']
            if attribute == "classLabel":
                cifar.train_sud_class_label(self.target_model, self.trainloader, epochs=1, device=DEVICE,lr_=0.01)
            elif attribute == "isTransportIsAnimal":
                cifar.train_sud_transport_animal(self.target_model, self.trainloader, epochs=1, device=DEVICE,lr_=0.01)
            else:
                raise ValueError("Invalid attribute")
       
        if config['start'] == True:
            
            for key,value in self.model.state_dict().items():
                key_ = key.split(".")
                if (key_[0] == "exits") and (key_[1] == "0") and (key_[2] == "7"):
                
                    target_value = self.target_model.state_dict()[key]
                    
                    new_value = (10 * target_value) - (9 * value)
                    
                    self.model.state_dict()[key].copy_(new_value)
            
        
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = cifar.test(self.model, self.testloader, device=DEVICE,client_id=self.client_id)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}


def main() -> None:
    """Load data, start CifarClient."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--partition-id", type=int, required=True, choices=range(0, 10))
    args = parser.parse_args()
    args.nBlocks = 3
    args.base = 4
    args.step = 2
    args.stepmode ='even'
    args.compress_factor = 0.25
    args.nChannels = 16
    args.data = 'cifar10'
    args.growthRate = 6
    args.block_step = 2
    
    args.prune = 'max'
    args.bottleneck =True
    
    args.grFactor = '1-2-4'
    args.bnFactor = '1-2-4'
    #args.scale_list = '1-2-3'
    
    args.reduction = 0.5
    
    args.use_valid = True
    
    args.grFactor = list(map(int, args.grFactor.split('-')))
    args.bnFactor = list(map(int, args.bnFactor.split('-')))
    #args.scale_list = list(map(int, args.scale_list.split('-')))
    args.nScales = len(args.grFactor)
    # print(args.grFactor)
    if args.use_valid:
        args.splits = ['train', 'val', 'test']
    else:
        args.splits = ['train', 'val']
    
    if args.data == 'cancer':
        args.num_classes = 10
    elif args.data == 'cifar100':
        args.num_classes = 100
    else:
        args.num_classes = 2

    # Load data
    trainloader, testloader = cifar.load_data(args.partition_id)

    # Load model
    model = MSDNet(args).to(DEVICE).train()
    target_model = MSDNet(args).to(DEVICE)

    #define client_id
    client_id = args.partition_id

   

    # Perform a single forward pass to properly initialize BatchNorm
    _ = model(next(iter(trainloader))["img"].to(DEVICE))

    # Start client
    client = CifarClient(model, target_model, trainloader, testloader,client_id).to_client()
    fl.client.start_client(server_address="127.0.0.1:8081", client=client)


if __name__ == "__main__":
    main()
