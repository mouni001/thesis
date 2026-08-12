"""Run literature-standard River baselines on the identical evolving stream."""

from __future__ import annotations

import argparse
import os
import pickle
import random
import time
from collections import Counter

import numpy as np
import torch
from river import forest, naive_bayes, tree

import evaluator_stream
from loaddatasets import loadinsects
from metrics_logger import StreamMetricLogger
from paths import data_path
from stream_annotations import get_stream_annotation
from train import PROTOCOL_REVISION, safe_output_name, select_contiguous_stream


def build_model(method: str, seed: int):
    method = method.strip().lower()
    if method == "hoeffding_tree":
        return tree.HoeffdingTreeClassifier()
    if method == "hoeffding_adaptive_tree":
        return tree.HoeffdingAdaptiveTreeClassifier(seed=seed)
    if method == "adaptive_random_forest":
        return forest.ARFClassifier(seed=seed, n_models=10)
    if method == "gaussian_naive_bayes":
        return naive_bayes.GaussianNB()
    raise ValueError(f"Unsupported River baseline: {method}")


def feature_dict(row: torch.Tensor, indices) -> dict:
    values = row.detach().cpu().numpy().reshape(-1)
    return {f"feature_{int(original)}": float(value) for original, value in zip(indices, values)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-method", required=True, choices=[
        "hoeffding_tree",
        "hoeffding_adaptive_tree",
        "adaptive_random_forest",
        "gaussian_naive_bayes",
    ])
    parser.add_argument("-DataName", default="insects")
    parser.add_argument("-insects_csv", required=True)
    parser.add_argument("-feature_protocol", default="feature_evolution")
    parser.add_argument("-feature_scenario", default="balanced")
    parser.add_argument("-shared_frac", type=float, default=0.5)
    parser.add_argument("-feature_seed", type=int, default=1314)
    parser.add_argument("-split_ratio", type=float, default=0.8)
    parser.add_argument("-split_index", type=int, default=-1)
    parser.add_argument("-T1", type=int, default=5000)
    parser.add_argument("-t", type=int, default=1000)
    parser.add_argument("-eval_window", type=int, default=500)
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("-output_name", required=True)
    parser.add_argument("-run_sanity_baselines", type=int, default=0)
    args, _ = parser.parse_known_args()

    if args.DataName.strip().lower() != "insects":
        raise ValueError("River baseline runner currently supports INSECTS only")
    random.seed(args.seed)
    np.random.seed(args.seed)

    x1, y1, x2, y2, feature_metadata = loadinsects(
        args.insects_csv,
        split_ratio=args.split_ratio,
        split_index=None if args.split_index < 0 else args.split_index,
        feature_protocol=args.feature_protocol,
        shared_frac=args.shared_frac,
        feature_seed=args.feature_seed,
        feature_scenario=args.feature_scenario,
        scaler_exclusion_size=max(0, int(args.T1 - args.t)),
        return_metadata=True,
    )
    t = min(args.t, len(x2))
    B = min(args.T1 - t, len(x1))
    T1 = B + t
    x1, y1, x2, y2 = select_contiguous_stream(x1, y1, x2, y2, B, t)
    y_stream = torch.cat([y1, y2]).view(-1).long()
    # Fix class identities from S1 only. Looking at S2 labels here would leak
    # future class frequencies into the definition of minority performance.
    classes, counts = np.unique(y1.numpy(), return_counts=True)
    min_class = int(classes[np.argmin(counts)])
    maj_class = int(classes[np.argmax(counts)])
    evaluator_stream.set_global_min_maj(min_class, maj_class)
    evaluator_stream.reset_stream_metrics(window=args.eval_window)
    num_classes = int(torch.max(y_stream).item()) + 1

    annotation = get_stream_annotation(args.insects_csv) or {}
    split_idx = int(feature_metadata["split_index"])
    stream_start = split_idx - B
    stream_end = split_idx + t
    known_local = [
        int(point - stream_start)
        for point in annotation.get("exact_abrupt_points", [])
        if stream_start <= point < stream_end
    ]
    reference_local = [
        int(point - stream_start)
        for point in annotation.get("reference_points", [])
        if stream_start <= point < stream_end
    ]
    output_name = safe_output_name(args.output_name)
    output_dir = data_path(output_name)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)

    metadata = {
        "method": args.method,
        "dataset": "insects",
        "insects_csv": args.insects_csv,
        "seed": int(args.seed),
        "T1": int(T1),
        "B": int(B),
        "t": int(t),
        "window_size": int(args.eval_window),
        "feature_transition_local": int(B),
        "minority_class": int(min_class),
        "majority_class": int(maj_class),
        "minority_reference": "evaluated_s1_segment_only",
        "dimension1": int(x1.shape[1]),
        "dimension2": int(x2.shape[1]),
        "feature_protocol": args.feature_protocol,
        "feature_scenario": args.feature_scenario,
        "feature_metadata": feature_metadata,
        "stream_start_original": int(stream_start),
        "transition_original": int(split_idx),
        "stream_end_original": int(stream_end),
        "stream_coordinate_system": "original_row",
        "stream_annotation": annotation,
        "known_abrupt_points_local": known_local,
        "reference_change_points_local": reference_local,
        "protocol_revision": PROTOCOL_REVISION,
        "output_name": output_name,
    }

    model = build_model(args.method, args.seed)
    logger = StreamMetricLogger(num_classes=num_classes)
    predicted = Counter()
    s1_indices = feature_metadata["s1_indices"]
    s2_indices = feature_metadata["s2_indices"]

    for step in range(T1):
        logger.start_step()
        if step < B:
            x = feature_dict(x1[step], s1_indices)
            y = int(y1[step].item())
        else:
            j = step - B
            x = feature_dict(x2[j], s2_indices)
            y = int(y2[j].item())

        predicted_proba = model.predict_proba_one(x)
        proba = np.zeros(num_classes, dtype=np.float32)
        for cls, probability in predicted_proba.items():
            cls = int(cls)
            if 0 <= cls < num_classes:
                proba[cls] = float(probability)
        if float(proba.sum()) <= 0:
            proba.fill(1.0 / num_classes)
        else:
            proba /= proba.sum()
        inference_seconds = time.perf_counter() - logger._step_t
        logger.update(y_true=y, y_proba=proba)
        predicted[int(np.argmax(proba))] += 1
        training_start = time.perf_counter()
        model.learn_one(x, y)
        training_seconds = time.perf_counter() - training_start
        logger.end_step(
            inference_seconds=inference_seconds,
            training_seconds=training_seconds,
        )

    logger.save_npz(os.path.join(output_dir, "metrics"), metadata=metadata)
    logger.stop()
    with open(os.path.join(output_dir, "final_model.pkl"), "wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[INFO] contiguous original stream interval: [{stream_start}, {stream_end})")
    print(f"[INFO] known exact abrupt points local={known_local}")
    print(f"[INFO] Predicted class distribution: {predicted}")
    print(f"[OK] River baseline done: {args.method}")


if __name__ == "__main__":
    main()
