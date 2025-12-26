# Final_MS_GNN: GNN-based Merge-and-Shrink Abstraction Learning

## Overview
This repository contains a framework for learning Merge-and-Shrink (M&S) abstraction strategies in automated planning using Graph Neural Networks (GNNs). It integrates with the Fast Downward planner to provide a reinforcement learning environment where a GNN agent learns to make merge decisions.

The framework supports:
- **Reinforcement Learning**: Uses Stable-Baselines3 and Gymnasium.
- **GNN Architectures**: Custom GCN-based models with attention mechanisms for state representation.
- **Planning Domains**: Benchmarks for various planning domains (e.g., Blocksworld, Logistics).
- **Comprehensive Evaluation**: Comparison against random baselines and Fast Downward's built-in strategies.

## Requirements
- Python 3.8+
- PyTorch >= 1.10.0
- PyTorch Geometric >= 2.0.0
- Stable-Baselines3 >= 1.6.0
- Gymnasium >= 0.27.0
- Fast Downward (included in `downward/` directory)
- Additional dependencies in `requirements.txt`

## Setup
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Final_MS_GNN
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Build Fast Downward** (if not already built):
   Follow instructions in `downward/README.md` or `downward/BUILD.md`.
   Typically:
   ```bash
   cd downward
   python build.py
   ```

## Project Structure
```text
Final_MS_GNN/
├── benchmarks/          # Planning domain and problem files (PPDL)
├── downward/            # Fast Downward planner source and builds
├── evaluation/          # Scripts for evaluating trained models
├── experiments/         # Experiment orchestration and configs
│   ├── configs/         # Experiment and curriculum configurations
│   ├── core/            # Core training and evaluation logic
│   └── runners/         # Runner classes for different experiment types
├── src/                 # Core framework source code
│   ├── communication/   # IPC logic with Fast Downward
│   ├── environments/    # Gymnasium-compatible M&S environments
│   ├── models/          # GNN architectures (GCN, Attention)
│   ├── rewards/         # Reward function definitions
│   └── utils/           # Common utilities
├── results/             # Default directory for experiment outputs
└── requirements.txt     # Python dependency list
```

## Usage

### Listing Available Experiments
To see the list of predefined experiment configurations:
```bash
python experiments/run_experiment.py --list
```

### Running an Experiment
To run a specific experiment (training + evaluation):
```bash
python experiments/run_experiment.py --exp blocksworld_exp_1
```

### Post-Training Analysis
To run a comprehensive post-training analysis on a completed experiment:
```bash
python experiments/run_post_training_analysis.py blocksworld_exp_1
```

### Custom Pipeline
For more control, use `run_full_experiment.py`:
```bash
python experiments/run_full_experiment.py --exp blocksworld_exp_1 --training-only
```

## Environment Variables
- `CUDA_VISIBLE_DEVICES`: Set to empty string `''` in some scripts to force CPU usage for consistency.
- `PROJECT_ROOT`: Internally managed but can be overridden if necessary.

## Scripts & Entry Points
- `experiments/run_experiment.py`: Main entry point for running experiments.
- `experiments/run_full_experiment.py`: Comprehensive pipeline for training and evaluation.
- `experiments/run_post_training_analysis.py`: Generates detailed reports and plots after training.
- `experiments/validate_pipeline.py`: Quick check to ensure the environment and model work.

## Tests
- TODO: Add information about unit tests and how to run them.
- Existing validation script: `python experiments/validate_pipeline.py`

## License
The Fast Downward planner included in this repository is licensed under the GPL-3.0 License. See `downward/LICENSE.md` for details.
The M&S GNN framework code in this repository: TODO: Add framework license.
