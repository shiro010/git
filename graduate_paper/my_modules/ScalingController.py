import torch
from typing import List
from torch.nn import functional as F

class ScalingController:
    def __init__(self, method="distribution", beta=1.0, temperature=1.0):
        self.method = method
        self.beta = beta
        self.temperature = temperature

    def compute_gammas(self, scores: List[float]) -> List[float]:
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        if self.method == "probabilistic":
            alpha_k = F.softmax(scores_tensor / self.temperature, dim=0)
            gamma_k = 1.0 - self.beta * alpha_k
        elif self.method == "distribution":
            if len(scores) > 1:
                mu = scores_tensor.mean()
                sigma = scores_tensor.std(unbiased=True)
                if sigma == 0: sigma = 1.0
                z_k = (scores_tensor - mu) / sigma
            else:
                z_k = torch.zeros_like(scores_tensor)
            w_k = torch.pow(F.relu(z_k), 2)
            gamma_k = 1.0 / (1.0 + self.beta * w_k)
        elif self.method == "log":
            log_scores = torch.log1p(F.relu(scores_tensor))
            max_val = log_scores.max()
            if max_val > 0:
                normalized_log = log_scores / max_val
            else:
                normalized_log = log_scores

            gamma_k = 1.0 / (1.0 + self.beta * normalized_log)
        elif self.method == "baseline":
            gamma_k = torch.ones_like(scores_tensor)
        elif self.method == "existing":
            gamma_k = torch.full_like(scores_tensor, fill_value=-1)

        else:
            gamma_k = torch.ones_like(scores_tensor)
        return gamma_k.tolist()