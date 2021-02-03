# Robust Imitation Learning with Co-pseudo-labeling (RIL-Co)
Source code of the RIL-Co algorithm in AISTATS 2021 paper titled "Robust Imitation Learning from Noisy Demonstrations" by Voot Tangkaratt, Nontawat Charoenphakdee, and Masashi Sugiyama.
This repository includes pytorch code and datasets used for experiments in the paper. 
The base of the code is a clone of https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail. 
The ACKTR algorithm and training flow are used as implemented for mujoco in the pytorch-a2c-ppo-acktr-gail repository. 
The main changes from the pytorch-a2c-ppo-acktr-gail repository are imitation learning (IL) algorithms.

## Requirements
Experiments were run with Python 3.6.9 and these packages:
* pytorch == 1.3.1
* numpy == 1.14.0
* scipy == 1.0.1
* gym == 0.10.5
* pybullet == 2.1.0

## Running experiments
Important files are 
* main.py - Script to run imitation learning experiments with RIL-Co and baseline adversarial IL algorithms.
* a2c_ppo_acktr/algo/ail.py - Script implementing RIL-Co and baseline adversarial IL algorithms excluding BC and VILD (Those are in bc.py and vild.py).
* a2c_ppo_acktr/algo/ail_utils.py - Script implementing loss functions for RIL-CO and baseline adversarial IL algorithms.
* main_bc.py - Script to run imitation learning experiments with a baseline behavior cloning (BC) algorithm. .
* my_enjoy.py - Script to visualize learned policies. Policy models trained by RIL-Co with seed=1 and ACKTR expert policy models are included in ./trained_models. This script is a slight edit of enjoy.py in pytorch-a2c-ppo-acktr-gail repository.
* save_traj.py - Script to collect demonstration datasets from expert policy models. The datasets used in experiments are already included in /imitation_data directory.
* ploy_ril.py - Script to plot figures in the paper. Results of experiments are included in /results_IL directory (as text files). (Running main.py will append new lines to those results files, and ploy_ril.py will read the new lines intead.)
* test_loss.py - Script to visualize the loss function.

To set algorithms, set argument --il_algo *algorithm_name*.
*algorithm_name* can be "ril_co", "ril", "ail", "airl", "fairl", "vild".
Except airl and fairl, it is very important to also set argument --ail_loss_type *loss_name*, where loss_name is one of "logistic", "unhinged", "apl".
For more details, see /a2c_ppo_acktr/arguments.py
Example commands:
To run RIL-Co with the AP loss, using a HalfCheeahBullet dataset with noise rate 0.4:
```
main.py --il_algo ril_co --ail_loss apl --env_name HalfCheetahBulletEnv-v0 --noise_prior 0.4 --algo acktr --num-process 32 --num-steps 20 --use-proper-time-limits
```

To run a baseline GAIL with the logistic loss with the same datasets:
```
main.py --il_algo ail --ail_loss logistic --env_name HalfCheetahBulletEnv-v0 --noise_prior 0.4 --algo acktr --num-process 32 --num-steps 20 --use-proper-time-limits
```

## Visualization of learned policies
These are visualized behavior of expert policies, (one of) non-expert policies, and policies learned by RIL-Co. These visualizations and visualizations of the other non-expert policies can be reproduced using my_enjoy.py
* Expert policies

![](/videos/HalfCheetahBulletEnv-v0_expert/HalfCheetahBulletEnv-v0_expert.gif "Expert HalfCheetahBullet")
![](/videos/AntBulletEnv-v0_expert/AntBulletEnv-v0_expert.gif "Expert AntBullet")

![](/videos/HopperBulletEnv-v0_expert/HopperBulletEnv-v0_expert.gif "Expert HopperBullet")
![](/videos/Walker2DBulletEnv-v0_expert/Walker2DBulletEnv-v0_expert.gif "Expert Walker2DBullet")

* (One of) Non-expert policies

![](/videos/HalfCheetahBulletEnv-v0_expertP0.4/HalfCheetahBulletEnv-v0_expertP0.4.gif "Non-expert HalfCheetahBullet")
![](/videos/AntBulletEnv-v0_expertP0.4/AntBulletEnv-v0_expertP0.4.gif "Non-expert AntBullet")

![](/videos/HopperBulletEnv-v0_expertP0.4/HopperBulletEnv-v0_expertP0.4.gif "Non-expert HopperBullet")
![](/videos/Walker2DBulletEnv-v0_expertP0.4/Walker2DBulletEnv-v0_expertP0.4.gif "Non-expert Walker2DBullet")

* Learned policies 

![](/videos/HalfCheetahBulletEnv-v0_rilco/HalfCheetahBulletEnv-v0_rilco.gif "RIL-Co HalfCheetahBullet")
![](/videos/AntBulletEnv-v0_rilco/AntBulletEnv-v0_rilco.gif "RIL-Co AntBullet")

![](/videos/HopperBulletEnv-v0_rilco/HopperBulletEnv-v0_rilco.gif "RIL-Co HopperBullet")
![](/videos/Walker2DBulletEnv-v0_rilco/Walker2DBulletEnv-v0_rilco.gif "RIL-Co Walker2DBullet")
