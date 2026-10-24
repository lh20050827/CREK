# CREK

This repository contains the training and evaluation code for **CREK**, an
unsupervised prompt-distillation method for CLIP. CREK first enriches the
frozen domain expert with learnable view heads, and then transfers the
resulting multi-view predictions to a lightweight student through instance-,
batch-, and class-level logit alignment. At inference time, only the student
CLIP and its learned textual prompt are used.

## Repository layout

```text
CREK/
├── clip/                         # CLIP implementation
├── configs/
│   ├── datasets/                 # dataset configurations
│   └── trainers/CREK/            # CREK configurations
├── Dassl.pytorch/dassl/          # bundled Dassl implementation
├── datasets/                     # base-to-novel dataset adapters
├── scripts/crek/                 # training and evaluation scripts
├── trainers/crek.py              # complete CREK trainer implementation
└── train.py                      # entry point
```

Data, pretrained CLIP files, domain-expert checkpoints, trained CREK
checkpoints, and experiment outputs are intentionally not included.

## Environment

The code was validated with Python 3.9, PyTorch 2.0.1, torchvision 0.15.2,
and NumPy 1.24.2.

```bash
conda create -n crek python=3.9 -y
conda activate crek
pip install -r requirements.txt
```

`train.py` loads the bundled Dassl package directly. Installing Dassl in
editable mode is optional:

```bash
pip install -e Dassl.pytorch --no-deps
```

## External assets

Set the following paths before running the scripts:

```bash
export CREK_DATA_ROOT=/path/to/datasets
export CREK_TEACHER_CLIP_PATH=/path/to/ViT-L-14.pt
export CREK_TEACHER_CHECKPOINT_ROOT=/path/to/base2new/teacher_model
export CREK_CROSS_DATASET_TEACHER_CHECKPOINT=/path/to/imagenet/model.pth.tar-20
```

Optional variables are `CREK_PYTHON`, `CREK_OUTPUT_ROOT`, `CREK_EPOCHS`, and
`CREK_WARMUP_EPOCHS`. Their defaults are `python`, a task-specific directory
under `output/`, 20 epochs, and 30 warm-up epochs, respectively.

For base-to-novel experiments, the teacher root is expected to contain one
checkpoint per dataset:

```text
${CREK_TEACHER_CHECKPOINT_ROOT}/
└── <DATASET.NAME>/
    └── VLPromptLearner/
        └── model-best.pth.tar
```

## Base-to-novel generalization

Training uses all unlabeled target-domain training images and automatically
evaluates the final epoch on novel classes:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/crek/base2new_train.sh dtd 1
```

Evaluate a saved final-epoch checkpoint separately:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/crek/base2new_test.sh dtd 1 base
CUDA_VISIBLE_DEVICES=0 bash scripts/crek/base2new_test.sh dtd 1 new
```

## Cross-dataset and domain generalization

Both settings use the ImageNet domain expert specified by
`CREK_CROSS_DATASET_TEACHER_CHECKPOINT`.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/crek/cross_dataset_train.sh dtd 1
CUDA_VISIBLE_DEVICES=0 bash scripts/crek/cross_dataset_test.sh dtd 1

CUDA_VISIBLE_DEVICES=0 bash scripts/crek/domain_generalization_train.sh imagenetv2 1
CUDA_VISIBLE_DEVICES=0 bash scripts/crek/domain_generalization_test.sh imagenetv2 1
```

Supported cross-dataset targets are Caltech101, DTD, EuroSAT, FGVCAircraft,
Food101, OxfordFlowers, OxfordPets, StanfordCars, SUN397, and UCF101. Supported
domain-generalization targets are ImageNetV2, ImageNet-Sketch, ImageNet-A, and
ImageNet-R.

## Checkpoints

CREK saves the student prompt and view-head states during training. Evaluation
scripts load the explicitly requested last epoch; the view heads are not used
for student inference.
