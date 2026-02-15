# train_driver.py
import os
import numpy as np
import torch
import torch.nn as nn

from loaddatasets import loadinsects
from mlp import MLP
from metrics_logger import StreamMetricLogger

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-3


def main():
    # 1) Stream
    x_S1, y_S1, x_S2, y_S2 = loadinsects()
    X_stream = torch.cat([x_S1, x_S2], dim=0).to(DEVICE)
    y_stream = torch.cat([y_S1, y_S2], dim=0).to(DEVICE)

    drift_start = len(x_S1)

    # 2) Model (6 classes)
    in_dim = X_stream.shape[1]
    model = MLP(in_planes=in_dim, num_classes=6).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 3) Logger
    logger = StreamMetricLogger(num_classes=6)
    logger.mark_drift(drift_start)

    # 4) Prequential loop (TEST then TRAIN)
    model.train()
    for i in range(len(X_stream)):
        x_i = X_stream[i].unsqueeze(0)  # [1, D]
        y_i = int(y_stream[i].item())

        logger.start_step()

        # forward ONCE (so we can train), but compute proba detached for logging
        logits_list = model(x_i)
        logits = logits_list[-1].squeeze(0)  # [C]

        proba = torch.softmax(logits, dim=-1).detach().cpu().numpy()  # (C,)
        logger.update(y_true=y_i, y_proba=proba)

        # TRAIN (same logits, same weights as test-time, then update)
        loss = criterion(logits.unsqueeze(0), torch.tensor([y_i], dtype=torch.long, device=DEVICE))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.end_step()

    # 5) Save
    save_dir = "./data/parameter_insects/metrics"
    logger.save_npz(save_dir)
    logger.stop()
    print(f"[OK] saved {os.path.join(save_dir, 'all_metrics.npz')}")


if __name__ == "__main__":
    main()
