# Robust Imitation Learning with Co-pseudo-labeling (RIL-Co)
This reposity contains PyTorch implementation of an imitation learning (IL) algorithm called Robust IL with Co-pseudo-labeling (RIL-Co) presented in AISTATS 2021 paper "Robust Imitation Learning from Noisy Demonstrations" by Voot Tangkaratt, Nontawat Charoenphakdee, and Masashi Sugiyama. 

## One line summary of the paper
The paper presented a theory and an algorithm for robustly learning an expert policy from noisy demonstrations containing both expert and non-expert demonstrations. 

## Requirements
Experiments in the paper were run with Python 3.6.9 and these packages:
* pytorch == 1.3.1
* numpy == 1.14.0
* scipy == 1.0.1
* gym == 0.10.5
* pybullet == 2.1.0

## Running experiments
The base of the code is a clone of the repository from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail. 
The ACKTR algorithm and training flow are as implemented in the pytorch-a2c-ppo-acktr-gail repository. 
The main changes from the pytorch-a2c-ppo-acktr-gail repository are IL algorithms implemented in a2c_ppo_acktr/algo/ail.py and a2c_ppo_acktr/algo/ril.py.

Important files for running experiments are 
* main.py - Script to run experiments with RIL-Co and comparison adversarial IL. (Script for behavior cloning (BC) is main_bc.py)
* a2c_ppo_acktr/algo/ril.py - Script implementing RIL-Co algorithm (Class RIL_CO). It also implements RIL-P used in the ablation study (Class RIL).

### Setting arguments
Setting arguments is necessary for running experiments via main.py (see /a2c_ppo_acktr/arguments.py)

* Use --il_algo *algorithm_name* to set algorithm. *algorithm_name* can be "ril_co", "ril", "ail", "airl", "fairl", "vild".
* Use --ail_loss_type *loss_name* to set loss function. *loss_name* can be "logistic", "unhinged", "apl".
* Use --env_name *environment_name* to set gym environment to train on. We tested RIL-Co on these PyBullet environments: "HalfCheetahBulletEnv-v0", "HopperBulletEnv-v0", "Walker2DBulletEnv-v0", "AntBulletEnv-v0". For these environments, the datasets used in paper are already included in /imitation_data directory. Generating datasets for other environments is possible by using save_traj.py and PyTorch policy models. 
* Use --noise_prior *noise_value* to set noise rate in the training dataset. *noise_value* can be 0.0, 0.1, 0.2, 0.3, 0.4. (0.0 means 100% expert data samples, while 0.4 means 60% of data samples come from expert policy and the rest 40% come from non-expert policies.)

Besides these arguments, you can also set rl algorithm and other hyper-parameters. Our experiments use ACKTR implemented by /ikostrikov/pytorch-a2c-ppo-acktr-gail with arguments --algo acktr --num-process 32 --num-steps 20 --use-proper-time-limits.

### Example commands and arguments
To run RIL-Co with the AP loss, using a HalfCheeahBullet dataset with noise rate 0.4:
```
main.py --il_algo ril_co --ail_loss apl --env_name HalfCheetahBulletEnv-v0 --noise_prior 0.4 --algo acktr --num-process 32 --num-steps 20 --use-proper-time-limits
```

To run GAIL with the logistic loss, using a HalfCheeahBullet dataset with noise rate 0.0:
```
main.py --il_algo ail --ail_loss logistic --env_name HalfCheetahBulletEnv-v0 --noise_prior 0.0 --algo acktr --num-process 32 --num-steps 20 --use-proper-time-limits
```

## Visualization of learned policies
These are visualized behavior of agents learned by RIL-Co, expert agents, and one of non-expert agents. Visualizations can be reproduced using my_enjoy.py. Quantitative results such as learning curves can be found in the paper. 

* Agents learned by RIL-Co from datasets with noise rate 0.4

![](/videos/HalfCheetahBulletEnv-v0_rilco/HalfCheetahBulletEnv-v0_rilco.gif "RIL-Co HalfCheetahBullet")
![](/videos/AntBulletEnv-v0_rilco/AntBulletEnv-v0_rilco.gif "RIL-Co AntBullet")

![](/videos/HopperBulletEnv-v0_rilco/HopperBulletEnv-v0_rilco.gif "RIL-Co HopperBullet")
![](/videos/Walker2DBulletEnv-v0_rilco/Walker2DBulletEnv-v0_rilco.gif "RIL-Co Walker2DBullet")

* Expert agents

![](/videos/HalfCheetahBulletEnv-v0_expert/HalfCheetahBulletEnv-v0_expert.gif "Expert HalfCheetahBullet")
![](/videos/AntBulletEnv-v0_expert/AntBulletEnv-v0_expert.gif "Expert AntBullet")

![](/videos/HopperBulletEnv-v0_expert/HopperBulletEnv-v0_expert.gif "Expert HopperBullet")
![](/videos/Walker2DBulletEnv-v0_expert/Walker2DBulletEnv-v0_expert.gif "Expert Walker2DBullet")

* One of non-expert agents

![](/videos/HalfCheetahBulletEnv-v0_expertP0.4/HalfCheetahBulletEnv-v0_expertP0.4.gif "Non-expert HalfCheetahBullet")
![](/videos/AntBulletEnv-v0_expertP0.4/AntBulletEnv-v0_expertP0.4.gif "Non-expert AntBullet")

![](/videos/HopperBulletEnv-v0_expertP0.4/HopperBulletEnv-v0_expertP0.4.gif "Non-expert HopperBullet")
![](/videos/Walker2DBulletEnv-v0_expertP0.4/Walker2DBulletEnv-v0_expertP0.4.gif "Non-expert Walker2DBullet")
