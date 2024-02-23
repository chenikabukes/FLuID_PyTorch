# Running FLuID with the Pytorch implementation

This builds on top of [Flower's Pytorch tutorial for running FL on Rapsberry Pis](https://github.com/adap/flower/tree/main/examples/embedded-devices). This is still based on Flower version 0.18.0

The implementation is split into two parts:
* Federated Dropout Strategy implementations
* Python scripts for running the experiments

There are separate implementations for each dataset (FEMNIST, CIFAR10, Shakespeare)
* Note CIFAR10 has 2 version, one for VGG16 and one for RESNET18


## Federated Dropout strategy implementations 

These implementations should placed in the actual flwr framework  (flwr/server/strategy)
### SETUP
1. After installing the flwr package, place the files in the folder `flwr_changed_files`(additional fedDrop strategy implementations and replace `aggregate.py`and `__init__.py`) in the installation path such as (`~/.local/lib/python3.8/site-packages/flwr/server/strategy`)
    * Note: the current fedDrop strategies are implemented based on the specific model architecture used for each dataset. Update the strategy implementations if the model architecture changes. 
2. Currently the dropout methods use a constant submodel size, with the value of `self.constant_pval`, this logic can be changed in the `aggregate_fit` method. 
3. The dropout methods can be changed in the `configure_fit` method 
    * `drop_rand` - random dropout
    * `drop_order` - ordered dropout
    * `drop_dynamic` - invariant dropout

## Python scripts for running the experiments

* Similar running FLuID on the mobile phones, there are `server.py` and a `client.py` files for running the experiments
* The instructions on running each experiment follows the general steps as the Android version implementation:

```bash
# launch your server. It will be waiting until one client connects
$ python server.py --server_address <YOUR_SERVER_IP:PORT> --rounds <NUM_ROUNDS< --min_num_clients <NUM_CLIENTS> --min_sample_size <NUM_CLIENTS_TO_SAMPLE> --model <MODEL_NAME>
```

```bash
# Launch each client seperately 
$ python3 client.py --server_address=<SERVER_ADDRESS>  --cid=<CLIENT_ID> --model=<MODEL_NAME> --device=<Device_to_run_on>
```

* Model name: 
    * `Net` for FEMNIST and CIFAR (VGG16) 
    * `ResNet18` for CIFAR (RESNET18)
    * `Shakespeare_LSTM` for Shakespeare
* Clients default run on the CPU, to run on the gpu specify `gpu` for the `--device`

* The run_`<dataset>`*.sh bash scripts allow initiating multiple clients on the same machine. Example PBS scripts are also included to initiate the server and clients. 
