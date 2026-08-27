import os

import torch


class Exp_Basic:
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if not self.args.use_gpu:
            print("Use CPU")
            return torch.device("cpu")

        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
        print(f"Use GPU: cuda:{self.args.gpu}")
        return torch.device(f"cuda:{self.args.gpu}")
