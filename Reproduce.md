### Install requirements
Run the following to install the requirements.
```
cd NeuReach
python3 -m pip install -r requirements.txt
```

Now you can follow the instructions below to reproduce the results in the paper.

### Results in Fig. 2 (Left) and Fig. 1
First, in the ```NeuReach``` directory, run the following command to train the NNs and evalaute them for each benchmark.
```
scripts/runall.sh
```
Once the above command finishes, you will see the results as in Fig. 2 (Left).

Next, run the following to generate the results in Fig. 1.
```
python3 scripts/plot_f16_single_timed.py --no_cuda --pretrained log/log_f16_GCAS_ours/checkpoint.pth.tar
python3 scripts/plot_jetengine.py --no_cuda --pretrained_ours log/log_jetengine_ours/checkpoint.pth.tar --pretrained_dryvr log/log_jetengine_dryvr/checkpoint.pth.tar
```
After it finshes, you should be able to find two images in the working directory: ```jetEngine.pdf``` and ```f16_GCAS.pdf```.

### Results in Fig. 2 (Right)
Run the following command to generate the results in Fig. 2 (Right).
```
python3 scripts/run_jetE_lambda.py
python3 scripts/plot_lambdas.py
```
After it finishes, you should be able to find an image in the working directory: ```lambdas.pdf```. Now we have reproduced all the results in the submission.
