# Developing a Lightweight Chess Agent for Aggressive Play using Reinforcement Learning

**Author:** Konstantinos Goutsias — Technical University of Crete (TUC)  
**Presented:** May 2026

> Chess engine combining MCTS with deep policy & value networks, trained via supervised learning on PGN databases and self-play. Features Syzygy tablebase integration for endgame play. Built with PyTorch as a thesis project.

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
  - [Data Representation](#data-representation)
  - [Neural Networks](#neural-networks)
  - [Monte Carlo Tree Search (MCTS)](#monte-carlo-tree-search-mcts)
  - [Training Pipeline](#training-pipeline)
- [Optimization](#optimization)
- [Aggressive Style Biasing](#aggressive-style-biasing)
- [Results](#results)
- [Setup & Usage](#setup--usage)
- [Requirements](#requirements)

---

## Overview

This project implements a lightweight AlphaZero-style chess engine that combines **Monte Carlo Tree Search (MCTS)** with deep **policy and value neural networks**. Unlike classical engines that rely on handcrafted heuristics, the agent learns entirely from data — first through supervised learning on grandmaster games, then through iterative self-play reinforcement learning.

The key research question addressed is:

> *Can we build a chess agent that is both lightweight and learns a specific playing style?*

Three agent variants were developed: **Baseline**, **Optimized**, and **Aggressive**.

---

## Motivation

AlphaZero (2017) demonstrated that a single neural network trained purely via self-play could surpass all previous chess engines. However, it required 5,000 TPUs for ~9 hours of training — entirely out of reach for academic research. Additionally, it optimizes purely for winning, with no regard for playing style.

This project addresses both limitations by:
- Building a resource-efficient pipeline that runs on consumer hardware (RTX 2070)
- Introducing aggressive style biasing mechanisms that shape *how* the agent plays, not just *whether* it wins

---

## Project Structure

```
├── engine/
│   ├── engine.py                         # Main UCI engine entry point
│   ├── mcts_game_parallel_aggressive.py  # Current best MCTS implementation
│   ├── self_play_game_parallel_aggressive.py  # Self-play data generation
│   ├── self_play_train.py                # Policy & value network training
│   ├── pipeline.py                       # Full training pipeline orchestration
│   ├── auxiliary_func.py                 # Board encoding, model utilities
│   ├── policy_network.py                 # Policy network definition
│   ├── value_network.py                  # Value network definition
│   ├── dataset.py                        # Dataset utilities
│   ├── syzygy.py                         # Syzygy tablebase interface
│   ├── test_network_vs_network.py        # Engine evaluation via Cutechess
│   ├── mcts_versions/                    # Archived MCTS implementations
│   │   ├── initial_mcts.py
│   │   └── mcts_tt_tb.py
│   └── self_play_versions/              # Archived self-play implementations
│       ├── initial_self_play.py
│       └── self_play.py
│   └── supervised_learning/
│       ├── train_policy_network.py
│       └── train_value_network.py
├── model/
│   ├── move_to_int                       # Move vocabulary (1892 moves)
│   ├── POLICY_MODEL_10EPOCHS_V0.pth      # Initial supervised model
│   ├── VALUE_MODEL_10EPOCHS_V0.pth
│   ├── baseline/                         # Baseline self-play models
│   ├── optimized/                        # Optimized pipeline models
│   └── aggressive/                       # Aggressive style models
├── data/
│   ├── opening_books/
│   │   ├── gambit_opening_book.bin       # Gambit opening book
│   │   └── gambit_opening_book_2.bin
│   ├── database/                         # PGN game database (not tracked)
│   ├── tablebase/                        # Syzygy tablebases (not tracked)
│   └── self_play_data/                   # Generated self-play data (not tracked)
└── requirements.txt
```

---

## Architecture

### Data Representation

Each board position is encoded as an **8×8×15 tensor**:

| Planes | Content |
|--------|---------|
| 0–5 | White pieces (Pawn, Knight, Bishop, Rook, Queen, King) |
| 6–11 | Black pieces (Pawn, Knight, Bishop, Rook, Queen, King) |
| 12 | Legal destination squares |
| 13 | Legal origin squares |
| 14 | Side to move |

Each plane is a binary 8×8 bitmap. The move vocabulary contains **1,892 unique moves** encoded as integers.

### Neural Networks

Both networks share the same input format and a lightweight architecture:

**Input:** `8 × 8 × 15` tensor → **2 Conv layers → 2 Fully Connected layers**

**Policy Network**
- Output: probability distribution over 1,892 legal moves
- Loss: Cross-Entropy (supervised) / KL Divergence (self-play)
- Training time: ~30 min (supervised, RTX 2070)

**Value Network**
- Output: scalar value `v ∈ [−1, 1]` estimating the outcome for the current player
- Loss: Mean Squared Error (MSE)
- Training time: ~9 min (supervised, RTX 2070)

**Supervised training hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Epochs | 10 |
| Learning rate | 1 × 10⁻⁴ |
| Batch size | 256 |
| Hardware | RTX 2070 |

Initial training data: ~3.5 million positions from [database.lichess.org](https://database.lichess.org).

### Monte Carlo Tree Search (MCTS)

The engine uses PUCT-based MCTS with four phases repeated for `n` simulations:

1. **Selection** — traverse the tree by maximizing PUCT at each node:

   ```
   PUCT(s, a) = Q(s, a) + c_puct · P(s, a) · √(ΣN(s,b)) / (1 + N(s,a))
   ```
   where `Q(s,a) = W(s,a) / N(s,a)` is the mean value, `P(s,a)` is the prior from the policy network, and `c_puct` is an exploration constant.

2. **Expansion** — add all children of the selected leaf node; initialize each child's prior `P(s, a) = πθ(a|s)` from the policy network.

3. **Evaluation** — estimate the leaf's value using the value network: `v = Vφ(s) ∈ [−1, 1]`

4. **Backpropagation** — propagate `v` up the path, flipping sign at each level (zero-sum game), updating `N(s,a)` and `W(s,a)`.

After `n` simulations, a move is sampled from the visit count distribution over the root's children.

### Training Pipeline

The full pipeline follows an iterative AlphaZero-style loop:

```
Supervised Learning (kickstart)
        ↓
Self-play Data Generation  →  Self-play Training
        ↑                            ↓
   old net = new net    ←   new net > old net?
```

Each pipeline iteration:
1. Generate self-play games using the current best network + MCTS
2. Store game data as `(board_state, π, z)` tuples where `π` is the MCTS visit distribution and `z` is the game outcome
3. Train policy network (KL divergence vs π) and value network (MSE vs z)
4. Evaluate new network vs old network over 500 games
5. If new network wins > 50%, promote it and start next iteration

---

## Optimization

The baseline pipeline took **~56 hours per iteration**. Several optimizations reduced this to **~3.5 hours (~20× speedup)**:

### MCTS-Level Optimizations

| Optimization | Samples/hr | Speedup |
|---|---|---|
| Baseline | 17,857 | — |
| Policy-based move pruning | 19,331 | +8.2% |
| Tablebase integration | 21,084 | +18.1% |
| Transposition table (50k cache) | 29,702 | +66.3% |
| Improved leaf parallelization (batch 128) | 43,682 | +144.7% |
| All combined (batch 128) | 49,086 | +174.8% |

**Policy-based move pruning:** During expansion, only actions where `πθ(a|s) > 0` are added to the tree — moves the policy considers impossible are pruned entirely.

**Transposition table:** A dictionary caches `(policy, value)` outputs for visited positions. Cache hits skip the neural network entirely (max size: 50,000 entries).

**Syzygy tablebase integration:** When ≤5 pieces remain on the board, the engine queries Syzygy tablebases for exact WDL (Win/Draw/Loss) results instead of running the value network. This gives perfect endgame play at lower computational cost.

**Improved leaf parallelization:** Instead of batching leaves from the same MCTS tree (which corrupts search statistics), multiple independent games are run simultaneously — one leaf per game is evaluated in a single batched neural network call. No virtual loss needed since trees are independent.

### Self-Play Level Optimizations

| Optimization | Samples/hr | Speedup |
|---|---|---|
| Baseline | 17,857 | — |
| Parallel workers | 62,112 | +248.0% |
| Opening book | 19,539 | +9.4% |
| Tablebase endgame | 20,090 | +12.5% |
| All combined | 71,116 | +300.0% |

**Combined (MCTS + Self-play) speedup: up to +1,862% (batch 128)**

Additional self-play optimizations include **tree reuse** (the subtree of the selected move is carried over to the next position, improving search quality at no cost) and an **opening book** (expert moves are followed for the first 5 moves, skipping MCTS entirely for the opening phase).

---

## Aggressive Style Biasing

Three levels of biasing were applied to shape the agent's playing style toward aggression:

### 1. Data-Level Biasing (Supervised Learning)
- The policy and value networks were **pretrained on gambit games** (Elo > 2400) sourced from the Lichess database
- This teaches the networks to value initiative, sacrifices, and open positions from the very start
- An **aggressive opening book** built from the same gambit dataset forces self-play into sharp, tactically imbalanced positions

### 2. MCTS-Level Biasing
- **Heuristic prior boost (Expansion):** Priors for checks and captures are boosted by a factor α > 1:
  ```
  P'(s, a) = P(s, a) · α,    α = 1.3
  ```
  This makes the tree search favor aggressive, tactical moves.

- **Depth-dependent discounting (Backpropagation):** Backed-up values are scaled by game depth:
  ```
  v_backup = γ^d · v,    γ = 0.97
  ```
  This discounts deep, slow victories and rewards quick, decisive wins.

### 3. Self-Play Level Biasing
- **Game-length reward shaping:** Win rewards are inversely proportional to game length, incentivizing faster wins:
  ```
  R' = R + λ · (1 / game_length),    λ = 0.5
  ```
- **Contempt parameter:** Draws are penalized with a negative reward:
  ```
  R_draw = -c,    c = 0.1
  ```
  This pushes the agent to prefer decisive results over draws.

**Final aggressive biasing parameters:** α=1.3, γ=0.97, λ=0.5, c=0.1

---

## Results

### Self-Improvement (Pipeline Testing — new vs old network, 500 games)

**Baseline Agent:**

| Pipeline Iteration | New Wins | Draws | Old Wins | Win % (cumulative vs baseline) |
|---|---|---|---|---|
| Iteration 1 | 129 | 280 | 91 | 58.6% |
| Iteration 2 | 157 | 244 | 99 | 61.3% |
| Iteration 3 | 163 | 237 | 100 | 62.0% |
| Iteration 4 | 168 | 234 | 98 | 63.2% |
| Iteration 5 | 166 | 243 | 91 | 64.6% |

**Optimized Agent:**

| Pipeline Iteration | New Wins | Draws | Old Wins | Win % (cumulative vs baseline) |
|---|---|---|---|---|
| Iteration 1 | 137 | 266 | 97 | 58.5% |
| Iteration 2 | 172 | 225 | 103 | 62.6% |
| Iteration 3 | 177 | 226 | 97 | 64.6% |
| Iteration 4 | 184 | 217 | 99 | 65% |
| Iteration 5 | 175 | 234 | 91 | 65.8% |

**Aggressive Agent:**

| Pipeline Iteration | New Wins | Draws | Old Wins | Win % (cumulative vs baseline) |
|---|---|---|---|---|
| Iteration 1 | 158 | 248 | 94 | 62.7% |
| Iteration 2 | 162 | 245 | 93 | 63.5% |
| Iteration 3 | 161 | 248 | 91 | 64.5% |
| Iteration 4 | 165 | 246 | 89 | 64.9% |
| Iteration 5 | 166 | 246 | 88 | 65.4% |

### Game Length Analysis (Move Threshold — % of games ending within N moves)

**Aggressive Agent:**

| Pipeline Iteration | ≤20 | ≤30 | ≤40 | >40 |
|---|---|---|---|---|
| Aggressive Iter 1 | 0.0% | 6.6% | 16.6% | 83.4% |
| Aggressive Iter 2 | 0.4% | 7.4% | 18.2% | 81.8% |
| Aggressive Iter 3 | 1.3% | 11.7% | 22.1% | 77.9% |
| Aggressive Iter 4 | 4.9% | 18.3% | 26.8% | 73.2% |
| Aggressive Iter 5 | 5.1% | 19.2% | 28.5% | 71.5% |

**Normal Agent:**

| Pipeline Iteration | ≤20 | ≤30 | ≤40 | >40 |
|---|---|---|---|---|
| Normal Iter 1 | 0.0% | 8.2% | 12.3% | 87.7% |
| Normal Iter 2 | 2.4% | 10.2% | 18.5% | 81.5% |
| Normal Iter 3 | 1.1% | 10.6% | 16.1% | 83.9% |
| Normal Iter 4 | 0.0% | 7.1% | 13.1% | 86.9% |
| Normal Iter 5 | 1.2% | 3.3% | 15.6% | 84.4% |

The aggressive agent consistently plays shorter, more decisive games.

### EAS-Score (Engine Aggressiveness Score)

A custom composite metric measuring playing aggressiveness:
- **EAS-Sacs:** frequency of material sacrifices (giving up material for positional/tactical compensation)
- **EAS-Shorts:** how quickly the engine finishes games

The aggressive agent scores higher on both components compared to the normal agent across all pipeline iterations.

### Final Evaluation (vs Torch800, 500 games via Cutechess)

| Engine | Opponent | W–D–L | Win% | Elo |
|---|---|---|---|---|
| Baseline | Torch800 | 151–248–101 | 59.9% | +27 ±3 |
| Aggressive Iter 5 | Torch800 | 191–230–79 | 70.7% | +64 ±13 |
| Aggressive Iter 5 | Baseline | 171–239–90 | 65.5% | +41 ±7 |

### Aggressive vs Normal Agent (500 games)

| Engine | Opponent | W–D–L | Win% | Elo |
|---|---|---|---|---|
| Aggressive | Normal | 167–184–149 | 53.4% | +10 ±3 |

Aggressive play creates practical tactical pressure, increasing the likelihood of opponent mistakes at this playing level.

---

## Setup & Usage

### Installation

```bash
pip install -r requirements.txt
```

### Download Models

Model weights (`.pth` files) are not tracked in this repository due to their size (~10MB each). Download them from the [Releases](../../releases) page and place them in the `model/` directory.

### Running the Engine

```bash
cd engine
python engine.py
```

The engine communicates via UCI protocol and can be connected to any UCI-compatible chess GUI (e.g., Arena, Cutechess, Lucas Chess).

### Running Self-Play Data Generation

```bash
cd engine
python self_play.py
```

### Running Self-Play Training

```bash
cd engine
python self_play_training.py
```

### Testing Networks Against Each Other

```bash
cd engine
python test_network_vs_network.py
```

---

## Requirements

```
chess==1.11.2
torch==2.8.0
numpy==2.0.2
tqdm==4.67.1
colorama==0.4.6
```

See `requirements.txt` for the full list.

---

## Future Work

- Residual network architectures (deeper, more expressive models)
- Advanced MCTS methods (e.g., Gumbel MCTS, improved exploration)
- Style experimentation: defensive play, human-like play
