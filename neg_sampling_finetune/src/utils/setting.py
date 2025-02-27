import random

import numpy as np
import torch


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# DataLoader에서 셔플사용하고 num_worker가 하나 보다 많을 때, 각각의 워커에 시드 설정을 해주지 않으면 재현성 보장이 안됨
def worker_init_fn(worker_id, seed):
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
