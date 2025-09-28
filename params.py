import argparse
from pathlib import Path
from datetime import datetime

def get_pangu_model_args():
    parser = argparse.ArgumentParser('Pangu', add_help=False)

    parser.add_argument('--num_air_variables', default=5, type=int)
    parser.add_argument('--num_surface_variables', default=5, type=int)

    parser.add_argument('--embedded_dim', default=16, type=int)
    parser.add_argument('--depths', default=(2,3,3,2), type=tuple)
    parser.add_argument('--heads', default=(2,2,2,2), type=tuple)
    parser.add_argument('--window_size', default=(2,6,6), type=tuple)
    parser.add_argument('--patch_size', default=(2,4,4), type=tuple)

    parser.add_argument('--drop_rate', default=0.2, type=float)
    parser.add_argument('--type_of_windows', default=(44,24,24,44), type=tuple)
     
    # Training parameters
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--finetune_epochs', default=10, type=int)
    parser.add_argument('--device', default="cuda", type=str)

    parser.add_argument('-g', '--gpuid', default=0, type=int, help="which gpu to use")
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--master_addr", default="127.0.0.1", type=str)
    parser.add_argument("--master_port", default="12356", type=str)
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
 
    # Model parameters
    parser.add_argument('--predict-steps', default=2, type=int, help='predict steps')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adamw"')
    # parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    # parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: None, use opt default)')
    # parser.add_argument('--clip-grad', type=float, default=1, metavar='NORM', help='Clip gradient norm (default: None, no clipping)')
    # parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=3e-6, help='weight decay (default: 0.05)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER', help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR', help='learning rate (default: 5e-4)')
    # parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct', help='learning rate noise on/off epoch percentages')
    # parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT', help='learning rate noise limit percent (default: 0.67)')
    # parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV', help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR', help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=0, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N', help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N', help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N', help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N', help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE', help='LR decay rate (default: 0.1)')

    # Pipline training parameters
    parser.add_argument('--pp_size', type=int, default=8, help='pipeline parallel size')
    parser.add_argument('--chunks', type=int, default=1, help='chunk size')

    return parser.parse_args()

def get_pangu_data_args(cur_proj): 
    data_args = {
        # 数据路径
        'root_path':            Path('/pangu_motion/'),
        'npy_path':             Path('/npy-data/'),
        'tp6hr_path':           Path('/tp6hr-data/'),

        # norm
        'norm_path':            f"/dataset/norms/norms.npy",
        'diff_path':            f"/dataset/norms/diff.npy",

        # log
        'train_log_path':       f"output/{cur_proj}/logs/pangu_train.log",
        'finetune_log_path':    f"output/{cur_proj}_finetune/logs/finetune_pangu_train.log",
        'test_log_path':        f"output/{cur_proj}/logs/pangu_test.json",

        # pt
        'latest_model_path':    f"output/{cur_proj}/ckpts/pangu_latest.pt",
        'finetune_model_path':  f"output/{cur_proj}_finetune/ckpts/finetune_pangu_latest.pt",
        'best_model_path':      f"output/pangu_motion/ckpts/pangu_best.pt",

        # 配置参数
        'train_start_datetime': datetime(2018, 1, 1, 1),  # 起始时间
        'train_end_datetime':   datetime(2022, 12, 31, 19),  # 结束时间
        'valid_start_datetime': datetime(2023, 1, 1, 1),
        'valid_end_datetime':   datetime(2023, 12, 31, 19),
        'test_start_datetime':  datetime(2018, 8, 19, 1),
        'test_end_datetime':    datetime(2018, 8, 31, 19),
        't_in':                 1,
        't_pretrain_out':       2,
        't_finetune_out':       8,
        't_final_out':          8,
    }
    return data_args
