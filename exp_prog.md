# Experiment Progress

## Sub-Optimality Gap Experiments

Based on arXiv:2508.01329 (Berseth et al.)

### MinAtar SpaceInvaders-v0

#### DQN (5M timesteps)

| Seed | Run ID | Status | Best Return | Mean Return | Optimality Gap |
|------|--------|--------|-------------|-------------|--------------|---------------|
| 1 | qjk3n697 | running | - | - | - |
| 2 | young-otter | running | - | - | - |
| 3 | calm-atlas | running | - | - | - |
| 4 | keen-bloom | running | - | - | - |

**Config:** `--total-timesteps 5000000 --seed {1-4} --track --log-dir runs/glen.berseth/`

**wandb project:** real-lab/sub-optimality

### Notes

- Optimality gap = best_experience - policy_performance
- The paper finds ~2-3x gap between best generated experience and learned policy
- Goal: reduce this gap

---

## Previous Runs

### PPO (500k timesteps, test)

- Run ID: tsbaq6v3 (completed)
- Config: `--total-timesteps 500000 --track`

### DQN Improvements (v2)

**Changes from baseline to reduce sub-optimality gap:**

- Learning rate: 2.5e-4 → **1e-4** (more stable updates)
- Exploration fraction: 0.25 → **0.50** (longer exploration)
- Buffer size: 100k → **500k** (more diverse experience)

**Hypothesis:** More exploration + slower learning + larger buffer = better exploitation of good experience