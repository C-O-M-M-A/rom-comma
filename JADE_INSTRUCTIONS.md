# Running the code on JADE II

Instructions for installing and running rom-comma code on JADE II. 

## Installation

When connecting to JADE II using SSH, you will first be placed in the login/head node. Software installation should be done
here as no internet access are available on the compute nodes (e.g. when running batch jobs).

Use the following commands pull `rom-comma` to your home directory:

```bash
# Navigate to home directory
cd ~

# Clone the rom-comma repository
git clone https://github.com/C-O-M-M-A/rom-comma.git

# Go inside the downloaded repository
cd rom-comma

# Load conda which we'll be using to install python packages
module load python/anaconda3

# Create a new conda virtual environment named rom-comma (name can be changed as you like) 
conda create -n rom-comma python=3.10

# Activate the rom-comma conda environment 
# (remember to change this name if you created an environment using a different name) 
source activate rom-comma

# Install python dependencies
pip install .

# Loaded conda modules on the head node interferes with 
# the environment in the compute node so I recommend just starting again from a fresh session
exit
```

## Testing installation and `gsa_from_function.py` script

To test our installation we will be running the script interactively. Connect to JADE II again then run:

```bash
# Gets an interactive session on the devel partition
run --partition=devel  --gres=gpu:1 --pty /bin/bash 
```

You will now be placed on a compute node which should have a GPU, running the following command:

```bash
nvidia-smi
```

should give an output similar to:

```bash    
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.142.00   Driver Version: 450.142.00   CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  On   | 00000000:06:00.0 Off |                    0 |
| N/A   35C    P0    42W / 163W |      0MiB / 32510MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

**Note: If the devel partition is busy, it's possible you will need to wait for some time until you get an interactive session**

We can now try running our `installation_test.py` script:

```bash
# When starting a new session we need to load 
# the modules and activate the conda environment
module load python/anaconda3
module load cuda/11.2
source activate rom-comma

# Run the test script
python installation_test.py
```

You should get an output similar to what's shown below:

```bash
2023-05-21 22:10:21.780894: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-05-21 22:10:36.916948: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
2023-05-21 22:11:30.152743: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1635] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 30935 MB memory:  -> device: 0, name: Tesla V100-SXM2-32GB-LS, pci bus id: 0000:06:00.0, compute capability: 7.0
Running Test using GPFlow(float='float64') on /CPU...
Running M=3, N=200, noise=0.04, is_noise_covariant=False, is_noise_variance_random=False, ext=0...
/jmain02/home/J2AD003/txk37/txk31-txk37/romcomma/rom-comma/romcomma/run/sample.py:52: UserWarning: 'centered' is deprecated and will be removed in SciPy 1.12. Please use 'scramble' instead. 'centered=True' corresponds to 'scramble=False'.
  return scipy.stats.qmc.LatinHypercube(M, centered=is_centered).random(N)
Running fold.0 gpr.v.a MOGP Regression took 0:00:07.
Running fold.0 gpr.v.a GSA2023-05-21 22:11:47.493700: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype int32 and shape [3,2]
	 [[{{node Placeholder/_0}}]]
2023-05-21 22:11:48.591487: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype int32 and shape [3,2]
	 [[{{node Placeholder/_0}}]]
2023-05-21 22:11:49.132839: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype int32 and shape [3,2]
	 [[{{node Placeholder/_0}}]]
 took 0:00:02.
...took 0:00:12.
...Running Test took 0:00:12.
```

We can then try to run the `gsa_from_function.py`:

```bash
# Runs the gsa_from_function.py script with the GPU device. 
# Output files will be saved to the outputs folder
# A tar file outputs.tar.gz will be created
python romcomma/user/gsa_from_function.py -d GPU -t outputs.tar.gz outputs
```

**Note: The device (-d) and tar (-t) flags are optional. Device defaults to CPU and a tar file is not 
generated without the -t flag.**

After running the above script, the output files should be located in a newly created `outputs` folder. The contents
of the `outputs` directory are also compressed into `outputs.tar.gz` file, ready to be downloaded. 


## Running as a job on JADE II

Interactive sessions are limited to the devel partition, you can only use a maximum of 1 GPU and there's a 
limited amount of running time, therefore it's only recommended for testing out your code. 
Submitting batch jobs is the recommended way to do any real computational work. 

### Creating and running a batch script

A batch script is a file that contains all of the steps for running your code, along with the resources that the jobs requires 
(e.g. the number of GPUs or how long the job will run). This section provides an example for running the `gsa_from_function.py` script.

Create a file in the repository's root directory with the following content, for this example we'll name it `run_gsa_from_function.sh`:

```bash
#!/bin/bash
# set partition of job (devel, small, big, or long)
#SBATCH --partition=small

# set number of GPUs
#SBATCH --gres=gpu:1

# set max wallclock time (HH:MM:SS)
#SBATCH --time=10:00:00

# Load the conda and cuda modules 
module load python/anaconda3
module load cuda/11.2

# Activate the conda environment
source activate rom-comma

# Runs the script
python romcomma/user/gsa_from_function.py -d GPU -t outputs.tar.gz outputs
```

The script can then be submitted using the command:

```bash
sbatch run_gsa_from_function.sh
```

You can check the status of your submitted script using the command:

```bash
squeue -u $USER
```

Which should result in an output similar to:

```bash
JOBID     PARTITION  NAME   USER      ST      TIME  NODES NODELIST(REASON)
123456     small run_gsa_ username    R       0:06      1 dgk707
```

The status column `ST` contains the current status of the job, `PD` means pending and `R` means the job is running.
More information about the output of `squeue` can be found in the [jade docs](https://docs.jade.ac.uk/en/latest/jade/scheduler/index.html#monitoring-jobs-with-the-command-squeue).


The text (std) output of the job can be read in a newly created file `slurm-[jobid].out`.

Once the job finishes running, it will disappear form the `squeue` output and in this particular case, an `outputs` folder and `outputs.tar.gz` file should be generated.
