# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
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
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

# ===================== load the reward module ===================== #
import sys
sys.path.insert(0, "/home/zeus/content/MinAtar")
sys.path.append("../")
from rllte.xplore.reward import RND, E3B
from minatar.gym import register_envs
register_envs()
# ===================== load the reward module ===================== #
import buffer_gap

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
    plot_freq: int = 10
    """The frequency of plotting"""
    wandb_project_name: str = "sub-optimality"
    """the wandb's project name"""
    wandb_entity: str = "real-lab"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "MinAtar/Asterix-v0"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 512
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    intrinsic_rewards: str = False
    """Whether to use intrinsic rewards"""
    top_return_buff_percentage: float = 0.05
    """The top percent of the buffer for computing the optimality gap"""
    return_buffer_size: int = 1000
    """the replay memory buffer size"""
    log_dir: str = "runs/glen.berseth/"
    """The directory to save the logs"""
    job_id : int = 0
    """The job id for the slurm job"""
    intrinsic_reward_scale: float = 1.0
    """The scale of the intrinsic reward"""
    old_wrappers: bool = False
    """Whether to use the old wrappers for the Atari environments"""
    num_layers: int = 1
    """The number of layers in the neural network"""
    num_units: int = 128
    """The number of units in the neural network"""
    use_layer_norm: bool = True
    """Whether to use layer normalization"""

def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        import buffer_gap
        env = buffer_gap.RecordEpisodeStatisticsV2(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()

        layers = [
                  nn.Flatten(),
                  layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), args.num_units)),
                  nn.Tanh()]
        for i in range(args.num_layers-1):
            layers.append(layer_init(nn.Linear(args.num_units, args.num_units)))
            layers.append(nn.Tanh())
            if args.use_layer_norm:
                layers.append(nn.LayerNorm(args.num_units))

        layers.extend([layer_init(nn.Linear(args.num_units, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)])
        self.critic = nn.Sequential(*layers)

        layers = [
            nn.Flatten(),
                  layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), args.num_units)),
                  nn.Tanh()]
        for i in range(args.num_layers-1):
            layers.append(layer_init(nn.Linear(args.num_units, args.num_units)))
            layers.append(nn.Tanh())
            if args.use_layer_norm:
                layers.append(nn.LayerNorm(args.num_units))

        layers.extend([
                layer_init(nn.Linear(args.num_units, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01)])

        self.actor = nn.Sequential(*layers)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    
    def get_action_deterministic(self, x):
        logits = self.actor(x)
        actions = torch.argmax(logits, dim=1)
        return actions


if __name__ == "__main__":
    args = tyro.cli(Args)
    # args.seed = int(os.environ.get("SLURM_PROCID", args.seed))
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    # jod_id = int(os.environ.get("SLURM_JOB_ID", 0)) * args.seed
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
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # ===================== build the reward ===================== #
    if args.intrinsic_rewards:
        klass = globals()[args.intrinsic_rewards]
        irs = klass(envs=envs, device=device, encoder_model="flat", obs_norm_type="none", beta=args.intrinsic_reward_scale)
    # ===================== build the reward ===================== #

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    #====================== optimality gap computation library ======================#
    import buffer_gap
    eval_envs = buffer_gap.SyncVectorEnvV2(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    gap_stats = buffer_gap.BufferGapV2(args.return_buffer_size, args.top_return_buff_percentage, agent, device, args, eval_envs)
    #====================== optimality gap computation library ======================#

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    last_global_step = global_step - args.plot_freq * 10000
    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())

            # ===================== watch the interaction ===================== #
            if args.intrinsic_rewards:
                irs.watch(observations=obs[step], actions=actions[step], 
                      rewards=rewards[step], terminateds=dones[step], 
                      truncateds=dones[step], next_observations=next_obs)
            # ===================== watch the interaction ===================== #

            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        gap_stats.add(info["episode"])
                        if global_step - last_global_step >= (args.plot_freq * 10000):
                            last_global_step = global_step
                            print(f"global_step={global_step}, episodic_return={info['episode']['r']}, iteration={iteration}")
                            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                            #====================== optimality gap computation logging ======================#
                            gap_stats.plot_gap(writer, global_step)
                            #====================== optimality gap computation logging ======================#


        # ===================== compute the intrinsic rewards ===================== #
        # get real next observations
        if args.intrinsic_rewards:
            real_next_obs = obs.clone()
            real_next_obs[:-1] = obs[1:]
            real_next_obs[-1] = next_obs

            intrinsic_rewards = irs.compute(samples=dict(observations=obs, actions=actions, 
                                                        rewards=rewards, terminateds=dones,
                                                        truncateds=dones, next_observations=real_next_obs
                                                        ))
            rewards += intrinsic_rewards
        # ===================== compute the intrinsic rewards ===================== #
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if iteration % args.plot_freq == 0:
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            # print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            #====================== log reward statistics ===================== #
            writer.add_scalar("charts/reward mean", rewards.mean(), global_step)
            writer.add_scalar("charts/reward top 95%", torch.mean(torch.topk(rewards.flatten(), min(500, rewards.numel()))[0]), global_step)
            writer.add_scalar("charts/return mean", rewards.mean(dim=0).mean(), global_step)
            # if torch.mean(torch.std(rewards, dim=0)) > 0:
            #     writer.add_scalar("charts/avg_reward_traj top 95%", torch.mean(torch.topk(rewards.mean(dim=0).flatten(), 2)[0]), global_step)
            if args.intrinsic_rewards:
                ## Here we iterate over the irs.metrics disctionary
                for key, value in irs.metrics.items():
                    writer.add_scalar(key, np.mean([val[1] for val in value]), global_step)
                    irs.metrics[key] = []


    envs.close()
    writer.close()
