# DSRM-HRL
Purifying Latent Preferences for Hierarchical Reinforcement Learning in Interactive Recommendation
## Code Structure

*   `code/model/agent/LHP1.py`: Implements the `LHP1` agent, which represents the core logic of the DSRM-HRL agent, including the training step and policy application.
*   `code/model/policy/OneStagePolicy_LHP1_d1.py`: Defines the hierarchical policy, including the high-level and low-level policies that generate actions (recommendation lists).
*   `code/train_actor_critic.py`: The main script used to train the reinforcement learning models.
*   `train_lhp_d1.sh`: A shell script for running the training process with a grid search over various hyperparameters.
*   `environment_dsrm_hrl.yml`: A Conda environment file to set up the required dependencies.

## Setup

1.  **Create Conda Environment**: Create and activate the Conda environment using the provided `environment_dsrm_hrl.yml` file.

    ```bash
    conda env create -f environment_dsrm_hrl.yml
    conda activate DSRM-HRL
    ```

2.  **Install Kernel (for Jupyter)**: If you plan to use Jupyter notebooks, install the environment as a kernel.
    ```bash
    python -m ipykernel install --user --name dsrm_hrl --display-name "dsrm_hrl"
    ```

## Training

To train the DSRM-HRL agent, you can use the `train_lhp_d1.sh` script. This script automates the process of running `train_actor_critic.py` with a comprehensive set of hyperparameters.

```bash
bash train_lhp_d1.sh
```

The script will iterate through different values for:
*   `FAIRNESS_WEIGHT`
*   `LIFECYCLE_WEIGHT`
*   `CRITIC_LR` (Critic Learning Rate)
*   `ACTOR_LR` (Actor Learning Rate)
*   `SEED`

Each training run will have its output and saved models stored in a unique directory under `output/KuaiRec/agents/`. The directory name is constructed from the hyperparameters of the run, for example: `LHP3_OneStagePolicy_LHP_gama_0.9_actor0.0001_critic0.001_niter20000_reg1e-05_ep0.01_noise0.1_bs128_epbs64_step20_seed11_0.05_0.2_econd-0.3_f0.1d(1+up)_wmlp_tau0.01_varlx`.


You can modify `train_lhp_d1.sh` to adjust the hyperparamer
