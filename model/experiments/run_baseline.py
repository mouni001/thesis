import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import numpy as np
import pandas as pd
import torch

import evaluator_stream
from feature_evolution_baselines import RIVER_METHODS, make_baseline
from loaddatasets import transition_location_info, loadadult, loadarrhythmia, loadcar, load_insects_from_csv, loadmagic, loadthyroid
from stream_recorder import StreamMetricLogger
from paths import data_path
from stream_annotations import get_stream_annotation
from train import PROTOCOL_REVISION, safe_output_name
from loaddatasets import select_contiguous_stream


def river_features(row, original_indices):
    values = row.detach().cpu().numpy().reshape(-1)
    return {
        f"feature_{int(original)}": float(value)
        for original, value in zip(original_indices, values)
    }


def load_paired_calibration(csv_path, feature_metadata, stream_start, calibration_size):
    frame = pd.read_csv(csv_path, header=None)
    features = frame.iloc[:, :-1].values.astype(np.float64)
    scaler_end = int(feature_metadata["scaler_fit_end_exclusive"])
    if scaler_end > int(stream_start):
        raise ValueError("Mapping calibration scaler overlaps evaluated observations")
    end = min(int(stream_start), scaler_end)
    start = max(0, end - int(calibration_size))
    if end <= start:
        raise ValueError("No historical prefix is available for paired calibration")
    mean = np.asarray(feature_metadata["scaler_mean"], dtype=np.float64)
    scale = np.asarray(feature_metadata["scaler_scale"], dtype=np.float64)
    standardized = (features[start:end] - mean) / np.where(scale == 0, 1.0, scale)
    old = standardized[:, np.asarray(feature_metadata["s1_indices"], dtype=int)]
    new = standardized[:, np.asarray(feature_metadata["s2_indices"], dtype=int)]
    return old, new, (start, end)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-method", choices=["fesl", "old3s", *sorted(RIVER_METHODS)], required=True
    )
    parser.add_argument("-baseline_hparams_json", default="{}")
    parser.add_argument("-DataName", default="insects")
    parser.add_argument("-insects_csv", default="")
    parser.add_argument("-feature_protocol", default="feature_evolution")
    parser.add_argument("-feature_scenario", default="balanced")
    parser.add_argument("-shared_frac", type=float, default=0.5)
    parser.add_argument("-feature_seed", type=int, default=1314)
    parser.add_argument("-split_ratio", type=float, default=0.8)
    parser.add_argument("-split_index", type=int, default=-1)
    parser.add_argument("-T1", type=int, default=1500)
    parser.add_argument("-t", type=int, default=1000)
    parser.add_argument("-eval_window", type=int, default=250)
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("-output_name", required=True)
    parser.add_argument("-mapping_calibration_size", type=int, default=500)
    parser.add_argument("-static_calibration_size", type=int, default=500)
    parser.add_argument("-protocol_revision", default=PROTOCOL_REVISION)
    parser.add_argument("-fesl_learning_rate", type=float, default=0.15)
    parser.add_argument("-fesl_hedge_beta", type=float, default=0.95)
    parser.add_argument("-fesl_mapper_ridge", type=float, default=1e-4)
    parser.add_argument("-transition_location", default="")
    parser.add_argument("-transition_fraction", type=float, default=None)
    parser.add_argument("-evaluation_region_start", type=int, default=None)
    parser.add_argument("-evaluation_region_end", type=int, default=None)

    args, _ = parser.parse_known_args()
    dataset = args.DataName.strip().lower()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if dataset == "insects":
        x1, y1, x2, y2, feature = load_insects_from_csv(
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
        t = min(int(args.t), len(x2))
        B = min(int(args.T1) - t, len(x1))
        T1 = B + t
        split_idx = int(feature["split_index"])
        stream_start = split_idx - B
        stream_end = split_idx + t
        x1, y1, x2, y2 = select_contiguous_stream(x1, y1, x2, y2, B, t)
        s1_indices = feature["s1_indices"]
        s2_indices = feature["s2_indices"]
        original_dimension = int(feature["original_dimension"])
        calibration_old = calibration_new = None
        calibration_interval = None
    elif dataset in {"magic", "adult", "arrhythmia", "car", "new-thyroid", "new_thyroid", "thyroid"}:
        loader = {
            "magic": loadmagic,
            "adult": loadadult,
            "arrhythmia": loadarrhythmia,
            "car": loadcar,
            "new-thyroid": loadthyroid,
            "new_thyroid": loadthyroid,
            "thyroid": loadthyroid,
        }[dataset]
        all_x1, all_y1, all_x2, all_y2 = loader(args.static_calibration_size)
        calibration_size = max(1, int(args.static_calibration_size))
        B = min(int(args.T1) - int(args.t), len(all_x1) - calibration_size)
        t = min(int(args.t), len(all_x2) - calibration_size - B)
        if B <= 0 or t <= 0:
            raise ValueError("Independent dataset is too short for calibration and evaluation")
        T1 = B + t
        calibration_old = all_x1[:calibration_size].numpy()
        calibration_new = all_x2[:calibration_size].numpy()
        x1 = all_x1[calibration_size : calibration_size + B]
        y1 = all_y1[calibration_size : calibration_size + B]
        x2 = all_x2[calibration_size + B : calibration_size + B + t]
        y2 = all_y2[calibration_size + B : calibration_size + B + t]
        split_idx = calibration_size + B
        stream_start = calibration_size
        stream_end = calibration_size + T1
        s1_indices = list(range(int(x1.shape[1])))
        s2_indices = list(range(int(x1.shape[1]), int(x1.shape[1] + x2.shape[1])))
        original_dimension = int(x1.shape[1] + x2.shape[1])
        calibration_interval = (0, calibration_size)
        feature = {
            "feature_protocol": "official_random_projection",
            "feature_scenario": "s2_expands",
            "original_dimension": original_dimension,
            "dimension1": int(x1.shape[1]),
            "dimension2": int(x2.shape[1]),
            "s1_indices": s1_indices,
            "s2_indices": s2_indices,
            "old_only_indices": s1_indices,
            "shared_indices": [],
            "new_only_indices": s2_indices,
            "old_only_count": len(s1_indices),
            "shared_count": 0,
            "new_only_count": len(s2_indices),
            "split_index": split_idx,
            "static_calibration_size": calibration_size,
            "scaler_fit_end_exclusive": calibration_size,
            "scaler_fit_source": "historical_calibration_prefix_only",
        }
    else:
        raise ValueError(f"Unsupported dataset for feature baseline: {dataset}")
    y_stream = torch.cat([y1, y2]).view(-1).long()
    classes, counts = np.unique(y1.numpy(), return_counts=True)
    minority = int(classes[np.argmin(counts)])
    majority = int(classes[np.argmax(counts)])
    evaluator_stream.set_global_min_maj(minority, majority)
    evaluator_stream.reset_stream_metrics(window=args.eval_window)
    num_classes = int(torch.max(y_stream).item()) + 1
    options = {
        "fesl_learning_rate": args.fesl_learning_rate,
        "fesl_hedge_beta": args.fesl_hedge_beta,
        "fesl_mapper_ridge": args.fesl_mapper_ridge,
        "river_options": json.loads(args.baseline_hparams_json),
    }
    model = make_baseline(
        args.method,
        int(x1.shape[1]),
        int(x2.shape[1]),
        original_dimension,
        num_classes,
        args.seed,
        **options,
    )
    if args.method in {"fesl", "old3s"}:
        if dataset == "insects":
            paired_old, paired_new, calibration_interval = load_paired_calibration(
                args.insects_csv,
                feature,
                stream_start,
                args.mapping_calibration_size,
            )
        else:
            paired_old, paired_new = calibration_old, calibration_new
        if args.method == "fesl":
            model.fit_mapping(paired_old, paired_new)

    annotation = get_stream_annotation(args.insects_csv) or {} if dataset == "insects" else {}
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
    output_dir = Path(data_path(output_name))
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    metadata = {
        "method": args.method,
        "baseline_hparams": (
            json.loads(args.baseline_hparams_json)
            if args.method in RIVER_METHODS
            else {
                key: value
                for key, value in vars(args).items()
                if key.startswith(args.method + "_")
            }
        ),
        "implementation_status": "protocol_adaptation" if args.method in {"fesl", "old3s"} else "river_implementation",
        "dataset": dataset,
        "insects_csv": args.insects_csv if dataset == "insects" else "",
        "seed": int(args.seed),
        "T1": int(T1),
        "B": int(B),
        "t": int(t),
        "window_size": int(args.eval_window),
        "feature_transition_local": int(B),
        "minority_class": minority,
        "majority_class": majority,
        "minority_reference": "evaluated_s1_segment_only",
        "dimension1": int(x1.shape[1]),
        "dimension2": int(x2.shape[1]),
        "feature_protocol": feature.get("feature_protocol", args.feature_protocol),
        "feature_scenario": feature.get("feature_scenario", args.feature_scenario),
        "feature_metadata": feature,
        "stream_start_original": int(stream_start),
        "transition_original": int(split_idx),
        "stream_end_original": int(stream_end),
        "stream_coordinate_system": (
            "original_row" if dataset == "insects" else "deterministic_dataset_sequence"
        ),
        "stream_annotation": annotation,
        "known_abrupt_points_local": known_local,
        "reference_change_points_local": reference_local,
        "protocol_revision": str(args.protocol_revision),
        "output_name": output_name,
        "mapping_calibration_interval_original": calibration_interval or [],
        "mapping_calibration_excluded_from_evaluation": True,
    }

    if args.method == "fesl":
        metadata.update({
            "method_label": "FESL (adapted)",
            "implementation_revision": "fesl_adapted_v2_both_classifiers_update",
            "source_algorithm": "FESL-c, Hou et al. (2017), Algorithm 2",
            "mapping_protocol": "historical_paired_prefix_before_evaluated_s1",
            "mapping_uses_labels": False,
            "mapping_calibration_size_actual": int(len(paired_old)),
            "mapping_fixed_during_evaluation": True,
            "s2_historical_classifier_updates": True,
            "adaptations": [
                "Historical paired calibration replaces transition-time overlap",
                "Ridge-regularized linear mapping",
                "Multiclass softmax classifiers and probability mixture",
                "Zero classifier initialization and local learning-rate/Hedge defaults",
            ],
        })

    if args.method == "old3s":
        from old3s_baseline import SOURCE_COMMIT
        metadata.update({
            "method_label": "OLD3S (adapted overlap)",
            "implementation_revision": "old3s_shallow_historical_overlap_v1",
            "source_commit": SOURCE_COMMIT,
            "source_algorithm": "Official OLD3S_Shallow",
            "mapping_protocol": "historical_paired_prefix_aligned_at_s1_end",
            "mapping_uses_labels": False,
            "mapping_calibration_size_actual": int(len(paired_old)),
            "baseline_hparams": {"latent_width": 1024, "heads": 5, "lr": 0.001,
                                 "beta": 0.9, "s": 0.008, "m": 0.99, "eta": -0.001},
            "adaptations": ["Unlabeled historical pairs replace labeled overlap",
                            "Multiclass cross-entropy on raw logits",
                            "MSE reconstruction for standardized inputs; SmoothL1 alignment",
                            "Pre-update predictions; per-head Hedge loss accumulation"],
        })

    metadata.update(transition_location_info(
        args, stream_start, stream_end, split_idx,
        feature.get("total_instances", len(x1) + len(x2))
    ))

    logger = StreamMetricLogger(num_classes=num_classes)
    start_wall = time.perf_counter()
    for step in range(T1):
        if step == B and args.method == "old3s":
            calibration_start = time.perf_counter()
            model.start_s2(paired_old, paired_new)
            metadata["alignment_calibration_seconds"] = time.perf_counter() - calibration_start
        logger.start_step()
        if step < B:
            raw = x1[step].numpy().astype(np.float64)
            y = int(y1[step].item())
            indices = s1_indices
            if args.method in RIVER_METHODS:
                sample = river_features(x1[step], indices)
                predicted = model.predict_proba_one(sample)
                proba = np.array(
                    [predicted.get(label, 0.0) for label in range(num_classes)],
                    dtype=np.float32,
                )
                if proba.sum() == 0:
                    proba.fill(1.0 / num_classes)
                else:
                    proba /= proba.sum()
            elif args.method in {"fesl", "old3s"}:
                proba = model.predict_s1(raw)
        else:
            j = step - B
            raw = x2[j].numpy().astype(np.float64)
            y = int(y2[j].item())
            indices = s2_indices
            if args.method in RIVER_METHODS:
                sample = river_features(x2[j], indices)
                predicted = model.predict_proba_one(sample)
                proba = np.array(
                    [predicted.get(label, 0.0) for label in range(num_classes)],
                    dtype=np.float32,
                )
                if proba.sum() == 0:
                    proba.fill(1.0 / num_classes)
                else:
                    proba /= proba.sum()
            elif args.method in {"fesl", "old3s"}:
                proba, historical, adaptive = model.predict_s2(raw)
        inference = time.perf_counter() - logger._step_t
        logger.update(y, proba)
        training_start = time.perf_counter()
        if args.method in RIVER_METHODS:
            model.learn_one(sample, y)
        elif step < B:
            model.learn_s1(raw, y)
        else:
            model.learn_s2(raw, y, historical, adaptive)
        training = time.perf_counter() - training_start
        logger.end_step(inference_seconds=inference, training_seconds=training)
    metadata["wall_clock_run_seconds"] = float(time.perf_counter() - start_wall)
    if args.method == "fesl":
        metadata["fesl_diagnostics"] = vars(model.diagnostics())
    logger.save_npz(str(output_dir / "metrics"), metadata=metadata)
    logger.stop()
    with (output_dir / "final_model.pkl").open("wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[OK] {args.method} completed on [{stream_start}, {stream_end})")


if __name__ == "__main__":
    main()
