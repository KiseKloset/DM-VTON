import numpy as np

import torch
from torch.nn.functional import adaptive_avg_pool2d

from metrics.pytorch_fid.inception import InceptionV3
from metrics.pytorch_fid.fid_score import calculate_frechet_distance


class PytorchFID:
    def __init__(self, dims=2048, device='cpu') -> None:
        self.dims = dims
        self.device = device
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.inception_model = InceptionV3([block_idx]).to(device)
        self.inception_model.eval()

    def compute_fid(self, data1, l1, data2, l2):
        m1, s1 = self.compute_activation_statistics(data1, l1)
        m2, s2 = self.compute_activation_statistics(data2, l2)

        fid = calculate_frechet_distance(m1, s1, m2, s2)
        return fid

    def compute_activation_statistics(self, data, length):
        act = self.accumulate_activation(data, length)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma
    
    def accumulate_activation(self, data, length):
        pred_arr = np.empty((length, self.dims))
        start_idx = 0

        for batch in data:
            batch = batch.to(self.device)
            with torch.no_grad():
                pred = self.inception_model(batch)[0]
            
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
            
            pred = pred.squeeze(3).squeeze(2).cpu().numpy()

            # Accumulate
            pred_arr[start_idx:start_idx + pred.shape[0]] = pred
            start_idx = start_idx + pred.shape[0]
    
        return pred_arr

    





