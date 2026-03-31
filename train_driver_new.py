import os
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

from loaddatasets import loadinsects
from mlp import MLP
from metrics_logger import StreamMetricLogger
import evaluator_stream

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-3


def _set_fixed_global_majority(y_stream):
    y_np = y_stream.detach().cpu().view(-1).numpy().astype(np.int64)
    classes, counts = np.unique(y_np, return_counts=True)
    maj_class = int(classes[np.argmax(counts)])
    min_class = int(classes[np.argmin(counts)])
    evaluator_stream.set_global_min_maj(min_class=min_class, maj_class=maj_class)
    return min_class, maj_class


def run_stream(model, X_stream, y_stream, drift_start, save_dir, train_until=None):
    """
    train_until:
      - None  => train forever (your current adaptive setup)
      - int   => train only for first train_until steps, then freeze (no-change baseline)
    """
    evaluator_stream.reset_stream_metrics()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    num_classes = int(torch.unique(y_stream.detach().view(-1)).numel())
    logger = StreamMetricLogger(num_classes=num_classes)
    logger.mark_drift(drift_start)

    model.train()

    for i in range(len(X_stream)):
        x_i = X_stream[i].unsqueeze(0)
        y_i = int(y_stream[i].item())

        logger.start_step()

        # TEST (forward)
        logits_list = model(x_i)
        logits = logits_list[-1].squeeze(0)

        proba = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        logger.update(y_true=y_i, y_proba=proba)

        # TRAIN (only if allowed)
        do_train = (train_until is None) or (i < int(train_until))
        if do_train:
            loss = criterion(
                logits.unsqueeze(0),
                torch.tensor([y_i], dtype=torch.long, device=DEVICE)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logger.end_step()

    os.makedirs(save_dir, exist_ok=True)
    logger.save_npz(save_dir)
    logger.stop()
    print(f"[OK] saved {os.path.join(save_dir, 'all_metrics.npz')}")


def run_global_majority_baseline(X_stream, y_stream, drift_start, save_dir):
    """Classical baseline: predict the full-stream global majority class at every step."""
    evaluator_stream.reset_stream_metrics()

    y_np = y_stream.detach().cpu().view(-1).numpy().astype(np.int64)
    num_classes = int(np.unique(y_np).size)

    classes, counts = np.unique(y_np, return_counts=True)
    global_majority = int(classes[np.argmax(counts)])

    logger = StreamMetricLogger(num_classes=num_classes)
    logger.mark_drift(drift_start)

    for i in range(len(X_stream)):
        y_i = int(y_stream[i].item())

        logger.start_step()

        proba = np.zeros(num_classes, dtype=np.float32)
        proba[global_majority] = 1.0
        logger.update(y_true=y_i, y_proba=proba)

        logger.end_step()

    os.makedirs(save_dir, exist_ok=True)
    logger.save_npz(save_dir)
    logger.stop()
    print(f"[OK] saved {os.path.join(save_dir, 'all_metrics.npz')} (global majority={global_majority})")


def run_cumulative_majority_baseline(X_stream, y_stream, drift_start, save_dir):
    """
    Prequential cumulative-majority baseline:
      predict using counts from past labels only, then update counts with current label.
    """
    evaluator_stream.reset_stream_metrics()

    y_np = y_stream.detach().cpu().view(-1).numpy().astype(np.int64)
    num_classes = int(np.unique(y_np).size)

    logger = StreamMetricLogger(num_classes=num_classes)
    logger.mark_drift(drift_start)

    seen_counts = Counter()
    default_class = int(np.min(y_np))

    for i in range(len(X_stream)):
        y_i = int(y_stream[i].item())

        logger.start_step()

        if len(seen_counts) == 0:
            pred_class = default_class
        else:
            pred_class = min(
                seen_counts.keys(),
                key=lambda c: (-seen_counts[c], c)
            )

        proba = np.zeros(num_classes, dtype=np.float32)
        proba[pred_class] = 1.0
        logger.update(y_true=y_i, y_proba=proba)

        # strictly prequential: update only after predicting/evaluating
        seen_counts[y_i] += 1

        logger.end_step()

    os.makedirs(save_dir, exist_ok=True)
    logger.save_npz(save_dir)
    logger.stop()
    print(f"[OK] saved {os.path.join(save_dir, 'all_metrics.npz')} (cumulative majority)")


def main():
    # 1) Stream
    x_S1, y_S1, x_S2, y_S2 = loadinsects()
    X_stream = torch.cat([x_S1, x_S2], dim=0).to(DEVICE)
    y_stream = torch.cat([y_S1, y_S2], dim=0).to(DEVICE)

    drift_start = len(x_S1)

    # Fixed global majority/minority used by evaluator for kappa_m and maj/min metrics
    min_class, maj_class = _set_fixed_global_majority(y_stream)
    print(f"[INFO] Global minority={min_class}, global majority={maj_class}")

    # 2) Model config
    in_dim = X_stream.shape[1]
    num_classes = int(torch.unique(y_stream.detach().view(-1)).numel())

    # -------------------------
    # A) Adaptive model (yours)
    # -------------------------
    model_adapt = MLP(in_planes=in_dim, num_classes=num_classes).to(DEVICE)
    save_dir_adapt = "./data/parameter_insects_adaptive/metrics"
    run_stream(
        model=model_adapt,
        X_stream=X_stream,
        y_stream=y_stream,
        drift_start=drift_start,
        save_dir=save_dir_adapt,
        train_until=None,  # train forever
    )

    # -------------------------
    # B) No-change baseline
    # -------------------------
    INIT_SIZE = 500
    model_nochange = MLP(in_planes=in_dim, num_classes=num_classes).to(DEVICE)
    save_dir_nochange = "./data/parameter_insects_nochange/metrics"
    run_stream(
        model=model_nochange,
        X_stream=X_stream,
        y_stream=y_stream,
        drift_start=drift_start,
        save_dir=save_dir_nochange,
        train_until=INIT_SIZE,
    )

    # -------------------------
    # C) Global majority baseline
    # -------------------------
    save_dir_global_majority = "./data/parameter_insects_global_majority/metrics"
    run_global_majority_baseline(
        X_stream=X_stream,
        y_stream=y_stream,
        drift_start=drift_start,
        save_dir=save_dir_global_majority,
    )

    # -------------------------
    # D) Cumulative majority baseline
    # -------------------------
    save_dir_cumulative_majority = "./data/parameter_insects_cumulative_majority/metrics"
    run_cumulative_majority_baseline(
        X_stream=X_stream,
        y_stream=y_stream,
        drift_start=drift_start,
        save_dir=save_dir_cumulative_majority,
    )


if __name__ == "__main__":
    main()
