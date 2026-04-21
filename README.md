# DeepRL Sub-Optimality

This repository focuses on reducing the **sub-optimality gap** in deep reinforcement learning - the phenomenon where RL policies only exploit about half of the good experience they generate (2-3x gap between best experience and learned performance).

Based on: https://arxiv.org/abs/2508.01329

## Project Goals

1. Measure the sub-optimality gap on benchmark tasks (MinAtar, Atari, etc.)
2. Develop methods to improve policy exploitation of good experience
3. Reduce the gap between best generated experience and learned performance

## Quick Start

### Installation

```bash
conda create -n cleanrl python=3.10
conda activate cleanrl
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### MinAtar Environments

```bash
https://github.com/Neo-X/MinAtar 
```

### Intrinsic Rewards

```bash
pip install git@github.com:Neo-X/rllte.git 
```

I did some edits to help with image formatting and size, logging, etc.

Reinstall the cleanrl environment to fix some package versions.

For installing 
```
pip install atari-py
```

Install cmake and zlib first
```
sudo apt-get install cmake zlib1g-dev
```

## Running Experiments

### MinAtar Space Invaders

```bash
cd cleanrl
uv run python cleanrl/ppo.py --env-id MinAtar/SpaceInvaders-v1 --total-timesteps 500000 --num-envs 4 --num-steps 128 --track
```

## Measuring Sub-Optimality

The key metric is the gap between:
- **Best experience generated**: Maximum return achieved during training
- **Policy performance**: Average return of the learned policy

See the paper above for the methodology.
