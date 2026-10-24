import os
import sys
from pathlib import Path

import argparse
import torch

REPO_ROOT = Path(__file__).resolve().parent
DASSL_ROOT = REPO_ROOT / "Dassl.pytorch"
if str(DASSL_ROOT) not in sys.path:
    sys.path.insert(0, str(DASSL_ROOT))

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import trainers.crek

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """Add the configuration variables required by CREK."""
    from yacs.config import CfgNode as CN

    # Keep generated few-shot pickle files outside the raw dataset tree when set.
    # Each dataset is stored in a separate DATASET.NAME namespace.
    cfg.DATASET.FEWSHOT_CACHE_ROOT = ""
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    # Fraction of the unlabeled training split retained for data-amount
    # ablations. The default keeps the original CREK data protocol unchanged.
    cfg.DATASET.UNLABELED_RATIO = 1.0

    # Config for independent Vision Language prompting (independent-vlp)
    cfg.TRAINER.IVLP = CN()
    cfg.TRAINER.IVLP.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.IVLP.N_CTX_TEXT = 2  # number of context vectors at the language branch
    cfg.TRAINER.IVLP.CTX_INIT = "a photo of a"  # initialization words (only for language prompts)
    cfg.TRAINER.IVLP.PREC = "fp16"  # fp16, fp32, amp
    # If both variables below are set to 0, 0, will the config will degenerate to COOP model
    cfg.TRAINER.IVLP.PROMPT_DEPTH_VISION = 9  # Max 12, minimum 0, for 0 it will act as shallow IVLP prompting (J=1)
    cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT = 9  # Max 12, minimum 0, for 0 it will act as shallow IVLP prompting(J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    
    cfg.TRAINER.MODAL = "base2novel"
    cfg.TRAINER.CREK = CN()
    cfg.TRAINER.CREK.N_CTX_VISION = 4  # number of teacher vision context vectors
    cfg.TRAINER.CREK.N_CTX_TEXT = 4  # number of teacher text context vectors
    cfg.TRAINER.CREK.CTX_INIT = ""  # teacher context initialization words
    cfg.TRAINER.CREK.PREC = "fp16"  # retained for configuration compatibility
    cfg.TRAINER.CREK.PROMPT_DEPTH_VISION = 9
    cfg.TRAINER.CREK.PROMPT_DEPTH_TEXT = 9
    cfg.TRAINER.CREK.PROJECT_LAYER = 2
    cfg.TRAINER.CREK.CE_WEIGHT = 0.0
    cfg.TRAINER.CREK.KD_WEIGHT = 1.0
    cfg.TRAINER.CREK.TEMPERATURE = 1.0
    cfg.TRAINER.CREK.TEACHER_NAME = "ViT/L-14"

    # External assets are intentionally not bundled with the source package.
    cfg.TRAINER.CREK.TEACHER_CLIP_PATH = ""
    cfg.TRAINER.CREK.TEACHER_CHECKPOINT_ROOT = ""
    cfg.TRAINER.CREK.CROSS_DATASET_TEACHER_CHECKPOINT = ""
    # When enabled, use a frozen pretrained CLIP with a fixed text template as
    # the teacher and do not load a downstream prompt checkpoint.
    cfg.TRAINER.CREK.ZERO_SHOT_TEACHER = False
    cfg.TRAINER.CREK.ZERO_SHOT_TEMPLATE = "a photo of a {}."

    # Angular-KD teacher-view augmentation. These defaults follow the paper,
    # while the semantic head loss is supplied by frozen-teacher pseudo labels
    # so CREK's unlabeled protocol is unchanged.
    cfg.TRAINER.CREK.ANGULAR = CN()
    cfg.TRAINER.CREK.ANGULAR.ENABLED = True
    cfg.TRAINER.CREK.ANGULAR.N_VIEWS = 5
    cfg.TRAINER.CREK.ANGULAR.DROPOUTS = [0.20, 0.25, 0.30, 0.35, 0.40]
    cfg.TRAINER.CREK.ANGULAR.MARGIN_INIT = 0.20
    cfg.TRAINER.CREK.ANGULAR.CONTRAST_T = 0.07
    cfg.TRAINER.CREK.ANGULAR.LOGIT_T = 4.0
    cfg.TRAINER.CREK.ANGULAR.FEATURE_INTER_WEIGHT = 0.10
    cfg.TRAINER.CREK.ANGULAR.FEATURE_INTRA_WEIGHT = 1.0
    cfg.TRAINER.CREK.ANGULAR.LOGIT_INTER_WEIGHT = 1.0
    cfg.TRAINER.CREK.ANGULAR.LOGIT_INTRA_WEIGHT = 1.0
    cfg.TRAINER.CREK.ANGULAR.SEMANTIC_WEIGHT = 0.80
    cfg.TRAINER.CREK.ANGULAR.WARMUP_EPOCHS = 30
    cfg.TRAINER.CREK.ANGULAR.WARMUP_BATCH_SIZE = 64
    cfg.TRAINER.CREK.ANGULAR.WARMUP_LR = 0.01
    cfg.TRAINER.CREK.ANGULAR.MAIN_LR = 0.01
    cfg.TRAINER.CREK.ANGULAR.WEIGHT_DECAY = 5e-4
    cfg.TRAINER.CREK.ANGULAR.MOMENTUM = 0.9

    # Required by the bundled optimizer implementation.
    cfg.OPTIM.RHO = 0.1
    cfg.OPTIM.EPS = 1e-8


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
