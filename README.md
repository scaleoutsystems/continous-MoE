# Online Continual Learning using Mixture-of-Experts (MoE)

This repository holds the experimental code for a Master's thesis project at Chalmers University of Technology with Scaleout Systems and AI Sweden. The goal of this project is to explore MoE and other models and see if they can be used for online continual learning.

## Getting Started

Experiments are driven by the `DIL_Experiment.ipynb` notebook which
reads a JSON configuration file describing dataset, partitioning,
model architecture, optimization details and miscellaneous hyperparameters.

### Configuration

Sample configurations can be found under `configs/example_config.json`.
Files use a JSONC style: lines starting with `//` (and /* … */ blocks) are
ignored by the loader, making it convenient to comment out options and
provide example blocks.  The reader function `experiment_fcns.load_config`
strips comments, prints a summary of key settings (including individual
seeds for dataset, model and training), ensures the dataset is available,
partitions data according to the chosen method (Dirichlet, static, etc.),
and returns instantiated objects such as the model, optimizer, loss,
dataloaders, replay buffer, and device.

Key fields in the config file include:

* `dataset`, `dataset_root`, `mini_dataset` – dataset selection.
* `num_partitions` and `partition` block – specify how to split training data.
* `train_frac`, `batch_size`, `shuffle` – loader parameters.
* `pretrain` – optional balanced pretraining set-up.
* `model` – name (e.g. `convnext_tiny`, `vit_moe`) and model-specific settings.
* `optimizer`, `scheduler`, `loss` – training settings.
* `epochs_per_domain` – number of epochs to run on each partition/domain.
* `replay`, `router_balancing`, `router_freeze_after_epochs` – continual learning options.

### Running an Experiment

1. Edit or duplicate a configuration JSON in `configs/`.
2. Open `DIL_Experiment.ipynb` and update `cfg_file` variable to point to
your config.
3. Execute the notebook. The results (metrics, confusion matrices) are
   saved under `logs/` (or the directory specified in the config).
4. Use `DIL_Metrics.plot_results` or `metrics_fcns.plot_expert_loads` to
   visualize outcomes.

### Utility Modules

* `dataset_fcns` – dataset construction, partitioning, loader creation and
  helper plotting functions.
* `models_fcns` – model factory for standard vision backbones and MoE ViT.
* `metrics_fcns` – parameter counting and expert-load visualizations.
* `experiment_fcns` – configuration reader and experiment setup logic.
* `data_analysis_fcns` – existing DIL logging and metrics utilities used by
  the notebook.

## License

Apache v2.0 License, please see `LICENSE`