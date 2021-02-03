import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')

    parser.add_argument(
        '--env_name',
        default="HalfCheetahBulletEnv-v0",
        help='environment to train on. RIL-Co is tested on ["HalfCheetahBulletEnv-v0", "AntBulletEnv-v0", "HopperBulletEnv-v0", "Walker2DBulletEnv-v0"].')

    ## arguments added for RIL-Co experiments. int is used instead of boolean and string for easier execution on gpu clusters. 
    parser.add_argument(
        '--il_algo', default='ail', help='il algorithm to use: ail, airl, fairl, vild, ril_co, ril. (ail is GAIL, ril is RIL-P.)' )
    parser.add_argument(
        '--il_lr', type=float, default=1e-3, help='learning rate discriminator (default: 1e-3)')
    parser.add_argument(
        '--ail_loss_type', default='apl', help='discriminator loss to use: logistic, unhinged, apl')
    parser.add_argument(
        '--ail_saturate', default=None, type=int, help='ail reward type: 1 (default) means max-min with reward=l(-g(x)), 0 means reward=g(x), -1 means non-saturate GAN with reward=-l(g(x))')
    parser.add_argument(
        '--reward_std', default=0, type=int, help='use standardized reward for IL. The original repo use this. This experiment does not use it to be consistent with the theory.')
    parser.add_argument(
        '--gp_lambda', default=10, type=float, help='gradient penalty regularization')
    parser.add_argument(
        '--noise_type', default="policy", help='type of noisy demonstrations. Chosen from ["policy", "action"]. "policy" means using sub-optimal policy snapshot for noisy demonstrations. "action" means adding noise to optimal action for noisy demonstrations.')
    parser.add_argument(
        '--noise_prior', type=float, default=0.0, help='noise proportion (noise rate). Chosen from [0, 0.1, 0.2, 0.3, 0.4]. It basically determines the number of noise state-action pairs.')
    parser.add_argument(
        '--traj_deterministic', type=int, default=1, help='Use demonstrations collected by deterministic policies')
    parser.add_argument(
        '--demo_sub_traj', type=int, default=1, help='sub sampling trajectory length 50, and use these sub-trajectories as demonstrations')
    parser.add_argument(
        '--plot_show', type=int, default=1, help='plot show')  # plot arguments for plot_il.py 
    parser.add_argument(
        '--plot_save', type=int, default=0, help='plot save')   
    parser.add_argument(
        '--plot_large', type=int, default=1, help='use large figure')  
    parser.add_argument(
        '--plot_each', type=int, default=1, help='plot learning curves for each noise rate')  

    ## arguments from the original repo (https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail). 
    ## Most are unchanged, except steps in eval-interval/save-interval/num-env-steps, and --gail is defaulted to True. Default rl algorithm is acktr instead of a2c. --env-name is moved to above. 
    parser.add_argument(
        '--algo', default='acktr', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument(
        '--gail',
        type=int, 
        default=1,
        help='do imitation learning with gail')
    parser.add_argument(
        '--gail-experts-dir',
        default='./gail_experts',
        help='directory that contains expert demonstrations for gail')
    parser.add_argument(
        '--gail-batch-size',
        type=int,
        default=128,
        help='gail batch size (default: 128)')
    parser.add_argument(
        '--gail-epoch', type=int, default=5, help='gail epochs (default: 5)')
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=16,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=5,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=32,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=100,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=15,   # 15 to get ~ 1000 lines for 10 million steps for acktr with 32 num process and 20 num steps.
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=20e6,   # 20million for IL experiments
        help='number of environment steps to train (default: 10e6)')
    # parser.add_argument(
    #     '--env-name',
    #     default='PongNoFrameskip-v4',
    #     help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument(
        '--log-dir',
        default='/tmp/gym/',
        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument(
        '--save-dir',
        default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=True,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    ## added for RIL-Co 
    args.is_atari = False

    return args
