import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch import nn

import decoders


def set_seed(seed=None, seed_torch=True):
    if seed is None:
        seed = np.random.choice(2 ** 32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    print(f'Random seed {seed} has been set.')
    return seed

def set_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("GPU is not enabled.")
    else:
        print("GPU is enabled.")

    return device

def seed_worker(worker_id):
    """
    DataLoader will reseed workers following randomness in
    multi-process data loading algorithm.

    Args:
        worker_id: integer
            ID of subprocess to seed. 0 means that
            the data will be loaded in the main process
            Refer: https://pytorch.org/docs/stable/data.html#data-loading-randomness for more details
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


DEVICE = set_device()
SEED = set_seed(seed=0, seed_torch=True)
RESULTS_PATH = 'results'


BEHAVIORAL_VARS = [
    'pv_x', 'pv_y', 'pv_speed', 'pv_dir', 'pv_dir_cos', 'pv_dir_sin',
]

partitions = [
    (
        [pd.DataFrame(np.random.rand(10, 100)) for i in range(100)],  # nsv_train
        np.random.rand(100, 10, 2),  # bv_train
        [pd.DataFrame(np.random.rand(10, 100)) for i in range(100)],  # nsv_val
        np.random.rand(100, 10, 2),  # bv_val
        [pd.DataFrame(np.random.rand(10, 100)) for i in range(100)],  # nsv_test
        np.random.rand(100, 10, 2),  # bv_test
    )
]


bins_before =  0 # bins of nsv prior to the output are used for decoding
bins_current = 1  # Whether to use concurrent time bin of neural data
bins_after = 0  # bins of nsv after the output are used for decoding

predict_bv = [BEHAVIORAL_VARS.index('pv_x'), BEHAVIORAL_VARS.index('pv_y')]
decoder_type = ['MLP', 'MLPOld', 'LSTM', 'M2MLSTM', 'NDT'][0]
cebra_model = None
clf = False

hyperparams = dict(
    batch_size=512*8,
    num_workers=5,  # https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/44
    model=decoder_type,
    model_args=dict(
        in_dim=None,
        out_dim=len(predict_bv),
        hidden_dims=[512, 512, .2, 512],
        args=dict(
            clf=clf,
            activations=nn.CELU,
            criterion=F.mse_loss if not clf else F.cross_entropy,
            epochs=10,
            lr=3e-2,
            base_lr=1e-2,
            max_grad_norm= 1.,
            iters_to_accumulate=1,
            weight_decay=1e-2,  # 0 for L1 regularization
            num_training_batches=None,
            scheduler_step_size_multiplier=2,
        )
    ),
    behaviors=predict_bv,
    bins_before=bins_before,
    bins_current=bins_current,
    bins_after=bins_after,
    device=DEVICE,
    seed=SEED
)

bv_preds_folds, bv_models_folds, norm_params_folds, metrics_folds = decoders.pipeline.train_model(
    partitions, hyperparams, resultspath=RESULTS_PATH, stop_partition=None)

