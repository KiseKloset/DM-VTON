import random
import torch
import torch.nn as nn
import numpy as np

import CVTON.models.models as models
from CVTON.config import get_test_arguments


def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


class CVTON(nn.Module):
    def __init__(self, device="cpu", checkpoint=None):
        super().__init__()
        self.opt = get_test_arguments()
        fix_seed(self.opt.seed)
        self.model = models.OASIS_model(self.opt)
        self.model.eval()
        self.device = device


    def forward(self, data_i):
        image, label = models.preprocess_input(self.opt, data_i, self.device)
        pred = self.model(image, label)
        return pred
    


'''
1. Introduction
- Overview
- Motivation
- Goal
- Summary (problems, methods, exp)
- Contributions (sorted by mức độ)

2. Related work
(abstract)
- Try on
- Recommendation

3. Proposed method
(abstract)
- Try on
- Recommendation

4. Experiments and Results
(abstract)
- Try on
- Recommendation

5. AR Application
(abstract)
- Application: How the application works

6. Discussion
(abstract)
- Limitation
- Downstream applications 

7. Conclusion
- Summarization:
- Future work





Paper: AR (abstract, introduction, related work, conclusion)

1. Introduction
- Figure: FID + params + FPS
- 3 first paragrpahs: shorter -> 2 paragraphs
- Longer motivation
- Contribution: realtime + augmentation

2. Related work
- Remove human representation

3. Proposed method
- Overview: summary Motivation (2, 3 sentences) + pipeline (teacher, student)
- Teacher network
- Studnet network
- Loss function (and others)
Small note: use yolo in video
- Pose Augmentation

4. Experiments
- Implementations
- Datasets
- Metrics
- Experiments
    - Compare with SOTA methods (add Publication)
        - Speed, FLOPs
        - FID
    - Analysis
- Human studying

5. Conclusion
'''