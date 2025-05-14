### Instructions to run:

You can directly run ipynb notebooks in cifar10 and imagenet directories for pruning respective pretrained models. 
Fine-grained pruning is implemented since it gave far better results.
Steps for channelwise pruning are also given in cifar10 notebooks but they aren't used in the experiments of this work.

main.py is the file containing code to quantize models channelwise based on our sensitivity metric.

## Training Configuration Options

### Core Parameters
| Argument      | Type   | Default     | Description                                         |
|---------------|--------|-------------|-----------------------------------------------------|
| `--model`     | str    | mobilenet   | Model architecture to train (`mobilenet`, etc.)     |
| `--dataset`   | str    | imagenet    | Dataset to use for training                         |
| `--lr`        | float  | 0.1         | Initial learning rate                               |
| `--lr_type`   | str    | exp         | Learning rate scheduler: `exp`/`cos`/`step3`/`fixed`|
| `--n_epoch`   | int    | 150         | Total number of training epochs                     |
| `--wd`        | float  | 4e-5        | Weight decay coefficient                            |

### Resource Management
| Argument      | Type   | Default     | Description                                         |
|---------------|--------|-------------|-----------------------------------------------------|
| `--n_gpu`     | int    | 1           | Number of GPUs to use                               |
| `--batch_size`| int    | 128         | Input batch size                                    |
| `--n_worker`  | int    | 4           | Number of workers for data loading                  |

### Paths & Seeds
| Argument      | Type   | Default     | Description                                         |
|---------------|--------|-------------|-----------------------------------------------------|
| `--data_root` | str    | None        | **Required** dataset path                           |
| `--seed`      | int    | None        | Random seed for reproducibility                     |

### Checkpoint & Evaluation
| Argument      | Type   | Default     | Description                                         |
|---------------|--------|-------------|-----------------------------------------------------|
| `--ckpt_path` | str    | None        | Path to checkpoint for resuming training            |
| `--eval`      | flag   | -           | Run evaluation only (no training)                   |

---

**Usage Example:**
python train.py
--model resnet50
--dataset imagenet
--data_root /path/to/imagenet
--lr 0.01
--batch_size 256
--n_epoch 100


**Important Notes:**
1. Must specify `--data_root` when using custom dataset locations.
2. Omit `--eval` flag for normal training mode.
3. For multi-GPU training, ensure `--n_gpu` matches available devices.
4. Pretrained models for Imagenet are directly obtained from the torchvision library. 
The same models are fine-tuned for CIFAR10 - you may change them as per requirements in the code - but note that models are still on the lab system and not on the git repo.