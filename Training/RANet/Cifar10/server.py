"""Flower server example."""
import torch
import cifar
from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics
from collections import OrderedDict
from RANet import RANet
import argparse
import os

args = argparse.ArgumentParser(description="Early Exit CLI")
args.nBlocks = 2
args.step = 4
args.stepmode ='even'
args.compress_factor = 0.25
args.nChannels = 16
args.data = 'cifar10'
args.growthRate = 6
args.block_step = 2

args.grFactor = '1-2-4'
args.bnFactor = '1-2-4'
args.scale_list = '1-2-3'
args.reduction = 0.5
args.use_valid = True

args.grFactor = list(map(int, args.grFactor.split('-')))
args.bnFactor = list(map(int, args.bnFactor.split('-')))
args.scale_list = list(map(int, args.scale_list.split('-')))
args.nScales = len(args.grFactor)

test_dl = None

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = RANet(args).to(DEVICE)
result_file_path = ("./save/best_model_epoch.txt")

#print(net)
#print(net.state_dict().keys())
#print(len(net.state_dict().keys()))
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        fraction_fit,
        min_fit_clients,
        min_available_clients,
        evaluate_metrics_aggregation_fn,
        #fit_metrics_aggregation_fn,
        on_fit_config_fn,
        testloader,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            #fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            on_fit_config_fn=on_fit_config_fn
        )
        self.testloader = testloader
    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ):
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        print("AGGREGATED METRICS", aggregated_metrics)

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            #print("Net",net.state_dict().keys())
            params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
            #for kk, vv in params_dict:
            #    print(vv.shape)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            #print(state_dict.keys())
            net.load_state_dict(state_dict, strict=True)

            # Save the model
            if os.path.exists("./save"):
                torch.save(net.state_dict(), f"./save/model_round_b_alexnet{server_round}.pth")
            else:
                os.mkdir("./save")
                torch.save(net.state_dict(), f"./save/model_round_b_alexnet{server_round}.pth")
            #loss, accuracy = cifar.test_global(net, self.testloader, device=DEVICE,client_id="Global Server")
            #print("GLOBAL MODEL ACCURACY ON TEST SET -->",accuracy)

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ):
        """Aggregate evaluation accuracy using weighted average."""

        if not results:
            return None, {}

        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        
        aggregated_accuracy = sum(accuracies) / sum(examples)
        print("aggregated_accuracy",aggregated_accuracy)
        print(f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}")

        
        if os.path.exists(result_file_path):
            with open(result_file_path, 'r') as fin:
                best_accuracy = float(fin.readlines()[-1].split(" ")[-2])
            fin.close()
            #print("FIN",accuracy)
            if aggregated_accuracy > best_accuracy:
                with open(result_file_path, 'a') as fin:
                    fin.write(f"model_round_b_alexnet{server_round}.pth aggregated_accuracy {aggregated_accuracy} \n")
                    # Save the model
                    torch.save(net.state_dict(), f"./save/model_b_alexnet_best.pth")
                fin.close()
                
        else:
            #os.mkdir(result_file_path)
            with open(result_file_path, 'w') as fin:
                fin.write(f"model_round_b_alexnet{server_round}.pth aggregated_accuracy {aggregated_accuracy} \n")
                print("writing for the first time")
            fin.close()
        

        

        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return aggregated_loss, {"accuracy": aggregated_accuracy}

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    #print("METRICS",metrics)
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


parser = argparse.ArgumentParser(description="Fedleaks")
parser.add_argument("--attribute", type=str, required=False, choices=["classLabel", "isTransportIsAnimal"], default="classLabel")
args = parser.parse_args()

def config_fn(round:int,attribute:str="classLabel") -> dict:
    if attribute != None:
        config = {'start':True,'round':round,'attribute':attribute}
    else:
        config = {'start':False,'round':round}
    return config
    
strategy = SaveModelStrategy(
    fraction_fit=1.0,
    min_fit_clients=2,
    min_available_clients=2,
    evaluate_metrics_aggregation_fn=weighted_average,
    on_fit_config_fn = config_fn,
    testloader = test_dl,

)

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8081",
    config=fl.server.ServerConfig(num_rounds=100),
    strategy=strategy,
    
)
