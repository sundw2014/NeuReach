## NeuReach: Learning Reachability Functions from Simulations

### Create the model file
The user needs to create a Python file describing the underlying system and providing a simulator. Examples can be found in the ```systems``` directory. Specifically, the user must define the following functions.
```python
def sample_X0(): # Produces a random initial set X0 from a distribution P1. The parameterized representation of X0 should be returned.
def sample_t(): # Produces a random sample of t from a distribution P2.
def sample_x0(X0): # Takes an initial set X0, and produces a random sample of x0 in X0 according to a distribution D(X0).
def simulate(x0): # Takes an initial state x0 and generates a finite trajectory which is a sequence of states at some time instants. The user should make sure that for every time instant returned by sample_t(), a state corresponding to it can be found in the simulated trajectory.
def get_init_center(X0): # Takes an initial set X0 and returns the mean value of the distribution D(X0).
def get_X0_normalization_factor(): # This function is optional. It returns the mean and std of the distribution for X0, which are then used to normalize the training data set and could make the training easier. You can simply return 0 for mean and 1 for std to avoid such a normalization.
```
You can include any ohter helper functions and constants in this file if needed. After finishing this file, name it as ```system_mysystem.py``` and put it into the ```systems``` directory.

### Usage
NeuReach provides the following command line user interface.
```text
usage: NeuReach.py [-h] [--system SYSTEM] [--lambda _LAMBDA] [--alpha ALPHA]
                   [--N_X0 N_X0] [--N_x0 N_X0] [--N_t N_T] [--layer1 LAYER1]
                   [--layer2 LAYER2] [--epochs EPOCHS] [--lr LEARNING_RATE]
                   [--data_file_train DATA_FILE_TRAIN]
                   [--data_file_eval DATA_FILE_EVAL] [--log LOG] [--no_cuda]

optional arguments:
  -h, --help            show this help message and exit
  --system SYSTEM       Name of the dynamical system.
  --lambda _LAMBDA      lambda for balancing the two loss terms.
  --alpha ALPHA         Hyper-parameter in the hinge loss.
  --N_X0 N_X0           Number of samples for the initial set X0.
  --N_x0 N_X0           Number of samples for the initial state x0.
  --N_t N_T             Number of samples for the time instant t.
  --layer1 LAYER1       Number of neurons in the first layer of the NN.
  --layer2 LAYER2       Number of neurons in the second layer of the NN.
  --epochs EPOCHS       Number of epochs for training.
  --lr LEARNING_RATE    Learning rate.
  --data_file_train DATA_FILE_TRAIN
                        Path to the file for storing the generated training
                        data set.
  --data_file_eval DATA_FILE_EVAL
                        Path to the file for storing the generated evaluation
                        data set.
  --log LOG             Path to the directory for storing the logging files.
  --no_cuda             Use this option to disable cuda, if you want to train
                        the NN on CPU.

```

For example, if you have put ```system_mysystem.py``` into the ```systems``` directory, you can simply run the following command to train the NN.
```bash
python3 NeuReach.py --no_cuda --system mysystem --log log_mysystem --data_file_train mysystem_train.pklz --data_file_eval mysystem_eval.pklz
```
After it finishes, the trained NN will be stored in ```log_mysystem/checkpoint.pth.tar```. To evaluate the trained model, you can run the following to print the volume and error.
```bash
python3 scripts/eval.py --no_cuda --pretrained log_mysystem/checkpoint.pth.tar --system mysystem
```
If you want to integrate the trained model in your own code, please see ```scripts/eval.py``` for how to load the trained model and get an estimate of the reachable set.

To reproduce the results in the paper, please refer to these [instructions](Reproduce.md).
