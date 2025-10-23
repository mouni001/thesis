# train_driver.py
import os
import numpy as np
import torch
import torch.nn as nn


from loaddatasets import loadinsects  # or loadmagic/loadadult/...
from mlp import MLP
from metrics_logger import StreamMetricLogger  # NEW

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-3

# 1) Stream
x_S1, y_S1, x_S2, y_S2 = loadinsects()
X_stream = torch.cat([x_S1, x_S2], dim=0).to(DEVICE)
y_stream = torch.cat([y_S1, y_S2], dim=0)

drift_start = len(x_S1)

# 2) Model
in_dim = X_stream.shape[1]
model = MLP(in_planes=in_dim, num_classes=2).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# 3) Logger
logger = StreamMetricLogger()
logger.mark_drift(drift_start)  # S1->S2 boundary

# 4) Prequential loop (TEST then TRAIN)
model.train()
for i in range(len(X_stream)):
    x_i = X_stream[i].unsqueeze(0)                 # [1, D]
    y_i = int(y_stream[i].item())

    logger.start_step()                            # start timing

    # TEST (before training)
    with torch.no_grad():
        logits = model(x_i)[-1].squeeze(0)         # last head
        p1 = torch.softmax(logits, dim=-1)[1].item()

    logger.update(y_true=y_i, y_proba1=p1)         # update metrics

    # TRAIN (on this instance)
    optimizer.zero_grad()
    loss = criterion(logits.unsqueeze(0), torch.tensor([y_i], dtype=torch.long, device=logits.device))
    loss.backward()
    optimizer.step()

    logger.end_step()                              # end timing/memory

# 5) Save
save_dir = "./data/parameter_insects/metrics"
logger.save_npz(save_dir)
print(f"[OK] saved {os.path.join(save_dir, 'all_metrics.npz')}")
logger.stop()
