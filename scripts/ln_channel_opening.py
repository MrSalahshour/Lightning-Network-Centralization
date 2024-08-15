from utils import load_data, make_agent, make_env, load_model
from stable_baselines3 import SAC, TD3, PPO
from numpy import load
import gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import random
import os
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.logger import TensorBoardOutputFormat

# from stable_baselines3.common.monitor import Monitor

def make_multiple_env(rank, data, env_params, seed):
    def _init():
        env = make_env(data, env_params, seed, multiple_env=True)
        env.seed(seed + rank)
        return env
    return _init


class SaveModelCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=1, reward_show = 2500):
        super(SaveModelCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.reward_show = reward_show

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_training_start(self):
        self._log_freq = 250  

        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f'model_{self.n_calls}_steps')
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Model saved at step {self.n_calls}")
        '''
        Log my_custom_reward every _log_freq(th) to tensorboard for each environment
        '''
        if self.n_calls % self._log_freq == 0:
            rewards = self.locals['rewards']
            self.tb_formatter.writer.add_scalar("Mean rewards",
                                                     np.mean(rewards),
                                                     self.n_calls)
            if self.verbose > 0:
                print("Mean Rewards: ",np.mean(rewards))

        return True
    
class EarlyStoppingCallback(BaseCallback):
    def __init__(self, check_freq: int, n_steps_without_progress: int, verbose=0):
        super(EarlyStoppingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.n_steps_without_progress = n_steps_without_progress
        self.best_reward = -np.inf
        self.n_steps_since_best_reward = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Get the current reward estimate
            x, y = self.model.ep_info_buffer.pop() #infos.get("episode")
            if y > self.best_reward:
                self.best_reward = y
                self.n_steps_since_best_reward = 0
            else:
                self.n_steps_since_best_reward += 1

            if self.n_steps_since_best_reward > self.n_steps_without_progress:
                print("Stopping training because the reward has not improved for {} steps.".format(self.n_steps_without_progress))
                return False
        return True

def train(env_params, train_params, tb_log_dir, tb_name, log_dir, seed):

    data = load_data(env_params['data_path'], env_params['merchants_path'], env_params['local_size'],
                    env_params['n_channels'],env_params['local_heads_number'], env_params["max_capacity"])
    
    # env = make_env(data, env_params, seed, multiple_env=True)
    num_cpu = 10  # Number of processes to use
    envs = SubprocVecEnv([make_multiple_env(i, data, env_params, seed+i) for i in range(num_cpu)])

    # model = make_agent(env, train_params['algo'], train_params['device'], tb_log_dir)
    model = make_agent(envs, train_params['algo'], train_params['device'], tb_log_dir)

    # model = load_model("PPO", env_params,"plotting/tb_results/trained_model/PPO_tensorboard_fixed_graph_50nodes_5lengthEpisode_mlp_complex_6featureVersion")
    # model.set_env(env)

    #Add Callback for early stopping
    # callback = EarlyStoppingCallback(check_freq=10, n_steps_without_progress=1000)
    # Define the callback
    save_freq = 100000  # Save every 100,000 steps
    save_path = r'C:\Users\user01\Downloads\Lightning-Network-Centralization\plotting\tb_results\trained_model\1_5mil'
    callback = SaveModelCallback(save_freq, save_path)

    # Train the model
    model.learn(total_timesteps=train_params['total_timesteps'], tb_log_name=tb_name, callback=callback, log_interval=10)

    # model.learn(total_timesteps=train_params['total_timesteps'], tb_log_name=tb_name)
    model.save(log_dir+tb_name)


def main():
    """
    amounts:   in satoshi
    fee_rate and fee_base:  in data {mmsat, msat}
    capacity_upper_scale bound:  upper bound for action range(capacity)
    maximum capacity:   in satoshi
    local_heads_number: number of heads when creating subsamples
    sampling_stage, sampling_k:    parameters of snowball_sampling
    """

    import argparse
    parser = argparse.ArgumentParser(description='Lightning network environment for multichannel')
    parser.add_argument('--algo', choices=['PPO', 'TRPO', 'SAC', 'TD3', 'A2C', 'DDPG'], required=True)
    parser.add_argument('--data_path', default='data/data.json')
    parser.add_argument('--merchants_path', default='data/merchants.json')
    parser.add_argument('--tb_log_dir', default='plotting/tb_results')
    parser.add_argument('--tb_name', required=True)
    parser.add_argument('--log_dir', default=r'C:/Users/user01\Downloads/Lightning-Network-Centralization/plotting/tb_results/trained_model/')
    parser.add_argument('--n_seed', type=int, default=1) # 5
    parser.add_argument('--total_timesteps', type=int, default=1500000)
    parser.add_argument('--max_episode_length', type=int, default=5)
    parser.add_argument('--local_size', type=int, default=50)
    parser.add_argument('--counts', default=[200, 200, 200], type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--amounts', default=[10000, 50000, 100000], type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--epsilons', default=[.6, .6, .6], type=lambda s: [float(item) for item in s.split(',')])
    parser.add_argument('--device', default='auto')
    parser.add_argument('--max_capacity', type = int, default=1e7) #SAT
    parser.add_argument('--n_channels', type=int, default=5)
    parser.add_argument('--mode', type=str, default='channel_openning')#TODO: add this arg to all scripts
    parser.add_argument('--capacity_upper_scale_bound', type=int, default=10)
    parser.add_argument('--local_heads_number', type=int, default=5)
    parser.add_argument('--sampling_k', type=int, default=4)
    parser.add_argument('--sampling_stages', type=int, default=4)

    

    
    args = parser.parse_args()

    train_params = {'algo': args.algo,
                    'total_timesteps': args.total_timesteps,
                    'device': args.device}

    env_params = {'mode' : args.mode,
                  'data_path': args.data_path,
                  'merchants_path': args.merchants_path,
                  'max_episode_length': args.max_episode_length,
                  'local_size': args.local_size,
                  'counts': args.counts,
                  'amounts': args.amounts,
                  'epsilons': args.epsilons,
                  'max_capacity': args.max_capacity,
                  'n_channels': args.n_channels,
                  'capacity_upper_scale_bound': args.capacity_upper_scale_bound,
                  'local_heads_number':args.local_heads_number,
                  'sampling_k':args.sampling_k,
                  'sampling_stages':args.sampling_stages}

    

    for seed in range(args.n_seed):
        train(env_params, train_params,
              tb_log_dir=args.tb_log_dir, log_dir=args.log_dir, tb_name=args.tb_name,
              seed=np.random.randint(low=0, high=1000000))

if __name__ == '__main__':
    main()
