import argparse
import json
import os
import random

import numpy as np
import torch

from exp.exp_long_term_forecasting import ExpLongTermForecast


# 数据集默认值：文件名、采样周期、变量数、批大小、学习率和扩散深度。
DATA_DEFAULTS = {
    "ETTh1": ("ETTh1.csv", 24, 7, 32, 0.008, 3, 3),
    "ETTh2": ("ETTh2.csv", 24, 7, 32, 0.008, 3, 3),
    "ETTm1": ("ETTm1.csv", 96, 7, 64, 0.015, 4, 3),
    "ETTm2": ("ETTm2.csv", 96, 7, 64, 0.015, 4, 3),
    "Weather": ("weather.csv", 144, 21, 32, 0.008, 3, 5),
    "Electricity": ("electricity.csv", 24, 321, 32, 0.008, 3, 20),
    "Traffic": ("traffic.csv", 24, 862, 32, 0.008, 3, 20),
    "Exchange": ("exchange_rate.csv", 7, 8, 32, 0.008, 3, 3),
    "Solar": ("solar_AL.txt", 144, 137, 32, 0.008, 3, 10),
    "Custom": (None, 24, None, 32, 0.008, 3, 3),
}


def build_parser():
    parser = argparse.ArgumentParser(description="DTSFormer long-term forecasting")
    parser.add_argument("--is_training", type=int, choices=(0, 1), default=1)
    parser.add_argument("--model_id", type=str, default="ETTh1_96_96")
    parser.add_argument("--random_seed", type=int, default=2021)

    parser.add_argument("--data", choices=tuple(DATA_DEFAULTS), default="ETTh1")
    parser.add_argument("--root_path", type=str, default="./data/")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--features", choices=("M", "S", "MS"), default="M")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--enc_in", type=int, default=None)
    parser.add_argument("--cycle_length", type=int, default=None)

    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--d_ff", type=int, default=128)
    parser.add_argument("--temporal_d_model", type=int, default=None)
    parser.add_argument("--spatial_d_model", type=int, default=None)
    parser.add_argument("--temporal_d_ff", type=int, default=None)
    parser.add_argument("--spatial_d_ff", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--tif_window", type=int, default=4)
    parser.add_argument("--tif_correction_scale", type=float, default=0.05)
    parser.add_argument("--tif_scale_floor", type=float, default=0.0)
    parser.add_argument("--node_dim", type=int, default=16)
    parser.add_argument("--graph_top_k", type=int, default=None)
    parser.add_argument("--graph_alpha", type=float, default=3.0)
    parser.add_argument("--graph_temperature", type=float, default=1.0)
    parser.add_argument("--graph_self_loop_weight", type=float, default=0.5)
    parser.add_argument("--gcn_depth", type=int, default=None)
    parser.add_argument("--propalpha", type=float, default=0.3)
    parser.add_argument("--diffusion_epsilon", type=float, default=0.1)
    parser.add_argument("--cross_diffusion_iters", type=int, default=3)
    parser.add_argument("--cross_diffusion_epsilon", type=float, default=0.3)

    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--lradj",
        choices=("type1", "type2", "3", "4", "5", "6", "constant", "TST"),
        default="type1",
    )
    parser.add_argument("--pct_start", type=float, default=0.3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")
    parser.add_argument("--output_dir", type=str, default="./outputs/")
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser


def prepare_args(args):
    (
        default_path,
        default_cycle,
        default_channels,
        default_batch,
        default_lr,
        default_depth,
        default_top_k,
    ) = DATA_DEFAULTS[args.data]
    args.data_path = args.data_path or default_path
    args.cycle_length = args.cycle_length or default_cycle
    args.batch_size = args.batch_size or default_batch
    args.learning_rate = args.learning_rate or default_lr
    args.gcn_depth = args.gcn_depth or default_depth
    args.graph_top_k = args.graph_top_k or default_top_k
    if args.enc_in is None:
        args.enc_in = 1 if args.features == "S" else default_channels

    if args.data_path is None or args.enc_in is None:
        raise ValueError("Custom data requires --data_path and --enc_in")
    if args.moving_avg <= 0 or args.moving_avg % 2 == 0:
        raise ValueError("--moving_avg must be a positive odd integer")
    if args.label_len > args.seq_len:
        raise ValueError("--label_len cannot exceed --seq_len")
    if args.tif_window < 0:
        raise ValueError("--tif_window must be non-negative")
    if args.tif_correction_scale < 0.0 or args.tif_scale_floor < 0.0:
        raise ValueError("TIF correction scales must be non-negative")
    if args.graph_top_k < 1:
        raise ValueError("--graph_top_k must be positive")
    if args.graph_temperature <= 0.0:
        raise ValueError("--graph_temperature must be positive")
    if not 0.0 <= args.graph_self_loop_weight <= 1.0:
        raise ValueError("--graph_self_loop_weight must be between 0 and 1")
    if not 0.0 <= args.propalpha <= 1.0:
        raise ValueError("--propalpha must be between 0 and 1")
    if args.diffusion_epsilon < 0.0:
        raise ValueError("--diffusion_epsilon must be non-negative")
    if not 0.0 <= args.cross_diffusion_epsilon <= 1.0:
        raise ValueError("--cross_diffusion_epsilon must be between 0 and 1")
    if args.cross_diffusion_iters < 1:
        raise ValueError("--cross_diffusion_iters must be positive")
    if args.accumulation_steps < 1:
        raise ValueError("--accumulation_steps must be positive")
    if args.spatial_d_model is not None:
        attention_dim = args.spatial_d_model
    else:
        attention_dim = args.d_model
    if attention_dim % args.n_heads != 0:
        raise ValueError("The spatial model dimension must be divisible by --n_heads")

    args.model = "DTSFormer"
    args.use_gpu = torch.cuda.is_available() and not args.cpu
    if args.use_amp and not args.use_gpu:
        raise ValueError("--use_amp requires CUDA")
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def experiment_name(args):
    return (
        f"{args.model_id}_DTSFormer_{args.data}_"
        f"sl{args.seq_len}_pl{args.pred_len}_seed{args.random_seed}"
    )


def main():
    args = prepare_args(build_parser().parse_args())
    seed_everything(args.random_seed)
    setting = experiment_name(args)
    checkpoint_dir = os.path.join(args.checkpoints, setting)
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(
        os.path.join(checkpoint_dir, "config.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(vars(args), file, indent=2)

    print(f"Experiment: {setting}")
    experiment = ExpLongTermForecast(args)
    if args.is_training:
        experiment.train(setting)
        experiment.test(setting)
    else:
        experiment.test(setting, load_checkpoint=True)
    if args.use_gpu:
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
