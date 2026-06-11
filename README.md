# Online Continual Learning using Mixture-of-Experts (MoE)

This repository holds the experimental code for a Master's thesis project at Chalmers University of Technology with Scaleout Systems and AI Sweden. The goal of this project is to explore MoE and other models and see if they can be used for online continual learning.

If you use this repository, please cite https://hdl.handle.net/20.500.12380/311205, "Mixture-of-Experts Architectures Through the Lens of Continual Learning" by Ian Coss MacLeod (2026). 

The models that were used for the thesis are: moe_vit.py (standard MoE in the thesis), vit_moe_imagelevel.py (variant 1 in the thesis), and switch_moe_vit.py (variant 2 in the thesis).

## Getting Started

Experiments are driven by the `DIL_Experiment.py` file which
reads a JSONC configuration file describing dataset, partitioning,
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

Note: All options possible can be seen in the various available config files. Additional options should be easy to implement following the factory pattern.

### Datasets

Note that you must have downloaded the datasets for running experiments and to properly point the config to the dataset. I used `datasets/` folder with the respective datasets such as officehome, cifar-10, and core50 within it. Only the datasets you use are needed, and the "Repository Architecture" section below includes how to add new datasets.

### Running an Experiment

1. Edit or duplicate a configuration JSONC in `configs/`.
2. Open `DIL_Experiment.py` and update `cfg_file` variable to point to
your config.
3. Execute the notebook. The results (metrics, confusion matrices) are
   saved under `results/` (or the directory specified in the config). Check the `logs/` directory for information on how an ongoing run is going.

Note: This can be run headless and checked in on later. Check `DIL_Experiment.py` for usage.

### Running Multiple Experiments

1. Edit or duplicate a configuration JSONC in `configs/`.
2. Create a metaconfig JSONC in `metaconfigs/`. Check the example metaconfig for details. Seeds are used for reproducability and fairness against runs of different models.
3. Open `batch_experiment.py` and update `metaconfig` variable to point to you metaconfig
4. Execute `batch_experiment.py` and check the `logs/` directory for information on how an ongoing run is going.

Note: This can be run headless and checked in on later. Check `batch_experiment.py` for usage.

### Results

When an experiment is completed, a results file is saved as a pickle (`.pt`). It is saved under the results folder selected in a folder of the name of the config. The name of the file is the name of the model type, the dataset used, and a timestamp. Example: vit_moe model run on CIFAR10, with timestamp `2023-01-01-12:34`. would be saved as `vit_moe_cifar10_01011234.pt`.

This file is used to view the results. There will also be a log file that explains the results by epoch and is updated as the experiment runs, so one can check the console and the log file to dissect what happened if errors occur and whether the experiments are all done.

Check your results folder (the name of the folder is configurable, but the default is `results/`) to ensure that you have the data from the run. 

The data saved in a results file does NOT include the final model weights. It includes per-epoch, per-partition, confusion matrices and R-matrices, which are used to calculate the rest of the metrics. If it is one of the MoE models, the expert usage history is also saved per domain and per epoch for later analysis. These files are used in the analysis notebook `DIL_results_analysis.ipynb`. They are generally around 1 KB per epoch for a given model.

### Analyzing Results

1. Select which configs' results you would like to compare and place them in a metaconfig. Results can be viewed per-file, per-config, or per-metaconfig where the all the results that used a given config are used to make population statistics and each config is compared against each other.
2. Set the desired config, results file, or meta-config in the results analysis notebook `DIL_results_analysis.ipynb`, in the second cell. Note that putting in a metaconfig will automatically select all configs in the metaconfig and override the `configs` variable. 
3. Execute the notebook cells of interest. The population statistics is for comparing all runs of a given config against the other configs selected within the metaconfig. The expert analysis only works on the MoE in the model factory that have been implemented. The expert usage analysis uses a specific results file, NOT the config or metaconfig. Confusion matrices and imbalanced learning metrics per epoch also only use a specific results file.


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

## Additional information

There are additional unfinished models in the `models_classes/` folder, but they are not yet tested or fully implemented. The models that were used for the thesis are: moe_vit.py, vit_moe_imagelevel.py (variant 1 in the thesis), and switch_moe_vit.py (variant 2 in the thesis). The hopevision.py, CMS_FFNN.py, cms_in_vit_depth.py, cms_in_vit_wide.py, swin_domainawarecms.py, multiscaletimescaleMoEViT.py, and multitimescaleFFNN.py are not yet fully implemented. pretrained_vit_proto_moe.py models are not yet tested. 

## Repository architecture

The repository is structured as follows:

 - `configs/`: JSONC experiment and metaconfig files. Copy and modify examples to run new experiments.
 - `datasets/`: Default dataset root. Place dataset folders here (see "Datasets and data layout" below).
 - `dataset_fcns/`: Dataset construction, partitioning, dataloader creation and helper plotting functions. Add new dataset loaders here and register them with the experiment factory.
 - `datasets/`: Supplementary or third-party dataset utilities used by the project.
 - `models_fcns` (module): Model factory that instantiates architectures based on the `model` field in a config.
 - `models_classes/`: Model architecture implementations (add new model classes here). Existing MoE/Vision Transformer implementations live here and can be used as templates.
 - `models/`: Additional model implementations, wrappers or pretrained-load utilities.
 - `experiment_fcns/`: Configuration reader and experiment setup logic (partitions, replay buffers, device setup, etc.).
 - `data_analysis_fcns/`: Logging, metrics and visualization helpers used by the notebooks.
 - `results/`, `logs/`: Output folders for run artifacts and logs.
 - `scripts/`, `batch_experiment.py`, `DIL_Experiment.py`: Runners and helpers for single or batched experiments.

### How to add new models

- Implement the architecture in `models_classes/` using an existing model (for example `moe_vit.py`) as a template. Keep the constructor signature and device handling consistent with other classes.
- Register the new model in the model factory (the `models_fcns` module) so the config `model: "your_model_name"` maps to a builder that returns an instantiated PyTorch model.
- Add any model-specific default settings to an example config in `configs/` and test via `DIL_Experiment.py`.

### Datasets and data layout (configuration)

- Default location: put datasets under the repository `datasets/` directory and set `dataset_root` in your config to `datasets/` (or point it to any other absolute path).
- Expected / supported layouts:
  - ImageFolder-style (recommended): `dataset_root/<name>/train/<class_name>/*.jpg` and `dataset_root/<name>/val/<class_name>/*.jpg` (works with torchvision-style loaders and helpers in `dataset_fcns`).
  - CIFAR-style: raw CIFAR files or the extracted `cifar-10-batches-py/` folder (this repo already contains `datasets/cifar-10-batches-py/`).
  - ImageNet-style: `<dataset_root>/imagenet/ILSVRC2012_img_train/<wnid>/*.JPEG` (or similar ILSVRC layout).
  - Custom single-file/CSV annotations: implement a loader in `dataset_fcns/` to parse annotations and create a PyTorch Dataset/Dataloader.
- Adding a new dataset:
  1. Place the raw data under `<dataset_root>/<your_dataset_name>/` following one of the layouts above.
  2. If the layout is non-standard, add a loader function to `dataset_fcns/` and hook it into the dataset selection logic (follow existing patterns in that module).
  3. In your config (`configs/your_config.jsonc`) set `dataset: "your_dataset_name"` and `dataset_root: "data/"` (or the absolute dataset path). Use `mini_dataset` for quick debug runs.
- Tips: keep an example config in `configs/` that documents expected `dataset_root` and `dataset` values for your dataset, and add a short README under `datasets/<your_dataset_name>/` describing the exact file layout used.

### Domain-incremental datasets (Core50, OfficeHome)

- Domain-incremental (a.k.a. domain/task-incremental) datasets contain the same set of classes observed across multiple domains or sessions (for example different capture sessions, backgrounds or domains). In this repository the dataset loaders for CORe50 and OfficeHome expose a `session_to_indices` mapping so the partitioner can split the data by whole domains/sessions instead of by class.

- CORe50 (expected layout)

  - The code expects CORe50 under: `<dataset_root>/core50_128x128_depth/` with session folders named `s1`, `s2`, ... (for example `s1..s11`). Each session directory contains object subfolders named `o1`, `o2`, ... and the image files inside those object folders.
  - The provided loader (`dataset_fcns/dataset_utils.py::Core50Dataset`) flattens selected session folders and builds a `session_to_indices` mapping keyed by integer session ids.
  - Example config snippet (JSONC):

    {
      // dataset loader name
      "dataset": "core50",
      "dataset_root": "data/",
      // optional: which sessions to include (defaults to all found)
      "settings": [1,2,3,4,5,6,7,8,9,10,11],
      "partition": { "type": "domainIncremental" },
      "num_partitions": 11
    }

- OfficeHome (expected layout)

  - OfficeHome should be placed under: `<dataset_root>/OfficeHomeDataset_10072016/` (or the nested layout `<...>/OfficeHomeDataset_10072016/OfficeHomeDataset_10072016/`); inside that folder there should be domain folders such as `Art`, `Clipart`, `Product`, `Real World`, each containing class subfolders and images.
  - The loader (`dataset_fcns/dataset_utils.py::OfficeHomeDataset`) builds a `session_to_indices` mapping where keys are domain names and selects domains according to `settings` if provided.
  - Example config snippet (JSONC):

    {
      "dataset": "officehome",
      "dataset_root": "data/",
      "settings": ["Art","Clipart","Product","Real World"],
      "partition": { "type": "domainIncremental" },
      "num_partitions": 4
    }

- Partitioning notes

  - When `partition.type` is set to `domainIncremental`, `experiment_fcns.load_config` will look for the dataset's `session_to_indices` mapping and split whole sessions/domains into partitions. The number of partitions (`num_partitions`) must divide the number of selected settings evenly (e.g., 4 domains -> `num_partitions` 1,2,4 are valid).
  - Use the `settings` field in the config to control which sessions/domains are included and their order (the order determines how domains are grouped into partitions).
  - This partitioning strategy is useful for experiments where each step/domain corresponds to a real-world change (new session, new domain) and you want to simulate continual arrival of domains.


