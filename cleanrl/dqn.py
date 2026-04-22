# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
# Limit threads for OpenBLAS
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
# Limit threads for MKL
os.environ["MKL_NUM_THREADS"] = "1"
# Limit threads for OpenMP (a common standard for parallel programming)
os.environ["OMP_NUM_THREADS"] = "1"
# Limit threads for VecLib (another potential backend)
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# Limit threads for NumExpr (if used for expression evaluation)
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import buffer_gap

# ===================== load the reward module ===================== #
import sys
sys.path.append("../")
from rllte.xplore.reward import RND
# ===================== load the reward module ===================== #

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    plot_freq: int = 1000
    """The frequency of plotting"""
    wandb_project_name: str = "sub-optimality"
    """the wandb's project name"""
    wandb_entity: str = "real-lab"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "MinAtar/Asterix-v1"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 500000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.10
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.50
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""
    intrinsic_rewards: str = "RND"
    """Whether to use intrinsic rewards"""
    top_return_buff_percentage: float = 0.05
    """The top percent of the buffer for computing the optimality gap"""
    return_buffer_size: int = 1000
    """The size of the return buffer for computing the optimality gap"""
    log_dir: str = False
    """The directory to save the logs"""
    job_id : int = 0
    """The job id for the slurm job"""
    intrinsic_reward_scale: float = 1.0
    """The scale of the intrinsic reward"""
    old_wrappers: bool = False
    """Whether to use the old wrappers for the Atari environments"""
    num_layers: int = 1
    """The number of layers in the neural network"""
    num_units: int = 120
    """The number of units in the neural network"""
    use_layer_norm: bool = False
    """Whether to use layer normalization"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        import buffer_gap
        env = buffer_gap.RecordEpisodeStatisticsV2(env)
        env.action_space.seed(seed)

        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()

        layers = [
                nn.Flatten(),
                  nn.Linear(np.array(envs.single_observation_space.shape).prod(), args.num_units),
                  nn.ReLU()]
        for i in range(args.num_layers-1):
            layers.append(nn.Linear(args.num_units, args.num_units))
            layers.append(nn.ReLU())
            if args.use_layer_norm:
                layers.append(nn.LayerNorm(args.num_units))

        layers.extend([nn.Linear(args.num_units, 84),
                        nn.ReLU(),
                        nn.Linear(84, env.single_action_space.n),])
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
    def get_action_deterministic(self, x):
        q_values = self.forward(x)
        actions = torch.argmax(q_values, dim=1)
        return actions


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

            poetry run pip install "stable_baselines3==2.0.0a1"
            """
        )
    args = tyro.cli(Args)
    # args.seed = int(os.environ.get("SLURM_PROCID", args.seed))
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.job_id}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            dir=args.log_dir+f"wandb/{run_name}"
        )
    if args.log_dir:
        writer = SummaryWriter(args.log_dir+f"runs/{run_name}", max_queue=1000)    
    else:
        writer = SummaryWriter(f"runs/{run_name}", max_queue=1000)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = buffer_gap.SyncVectorEnvV2(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # ===================== build the reward ===================== #
    if args.intrinsic_rewards:
        from rllte.xplore.reward import RND, E3B
        klass = globals()[args.intrinsic_rewards]
        irs = klass(envs=envs, device=device, encoder_model="flat", obs_norm_type="none", beta=args.intrinsic_reward_scale)
    # ===================== build the reward ===================== #
    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    #====================== optimality gap computation library ======================#
    import buffer_gap
    eval_envs = buffer_gap.SyncVectorEnvV2(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    gap_stats = buffer_gap.BufferGapV2(args.return_buffer_size, args.top_return_buff_percentage, q_network, device, args, eval_envs)
    #====================== optimality gap computation library ======================#

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    return_ = 0
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        # print("actions", actions)
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        return_ += rewards
        infos["return"] = return_

        # ===================== watch the interaction ===================== #
        if args.intrinsic_rewards:
            irs.watch(observations=obs, actions=actions, 
                    rewards=rewards, terminateds=terminations, 
                    truncateds=truncations, next_observations=next_obs)
        # ===================== watch the interaction ===================== #

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    gap_stats.add(info["episode"])
                    if global_step % args.plot_freq == 0:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        #====================== optimality gap computation logging ======================#
                        gap_stats.plot_gap(writer, global_step)
                        #====================== optimality gap computation logging ======================#

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                rewards_ = data.rewards
                # ===================== compute the intrinsic rewards ===================== #
                # get real next observations
                if args.intrinsic_rewards:
                    
                    intrinsic_rewards = irs.compute(samples=dict(observations=data.observations*1.0, actions=data.actions, 
                                                                rewards=data.rewards, terminateds=data.dones,
                                                                truncateds=data.dones, next_observations=data.next_observations*1.0
                                                                ))
                    rewards_ += intrinsic_rewards
                # ===================== compute the intrinsic rewards ===================== #
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations *1.0).max(dim=1)
                    td_target = rewards_.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations * 1.0).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % args.plot_freq == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    # print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    data_ = rb.sample(10000)
                    if global_step % 1000 == 0 and data_.rewards.shape[0] >= 10000:
                        writer.add_scalar("charts/rewards mean", data_.rewards.mean(), global_step)
                        writer.add_scalar("charts/rewards top 95%", torch.mean(torch.topk(data_.rewards.flatten(), 500)[0]), global_step)
                        # writer.add_scalar("charts/returns top 95%", torch.mean(torch.topk(data_.returns.flatten(), 500)[0]), global_step)
                    if args.intrinsic_rewards:
                        ## Here we iterate over the irs.metrics disctionary
                        for key, value in irs.metrics.items():
                            writer.add_scalar(key, np.mean([val[1] for val in value]), global_step)
                            irs.metrics[key] = []


                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.dqn_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=0.05,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
