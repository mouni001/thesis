# train.py
import argparse
import os
import random
import re
import torch
import torch.nn as nn
import numpy as np
from collections import Counter

import evaluator_stream
from metrics_logger import StreamMetricLogger
from loaddatasets import loadmagic, loadinsects
from mlp import MLP
from model import OLD3S_Shallow
from paths import data_path
from stream_annotations import get_stream_annotation


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROTOCOL_REVISION = "thesis_protocol_2026-08-11_v7"


def set_global_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_run_tag(value: str) -> str:
    """Return a filesystem-safe experiment tag without path traversal."""
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return tag.strip("._-")


def safe_output_name(value: str) -> str:
    """Validate a relative output path beneath model/data."""
    raw = str(value).strip().replace("\\", "/")
    if not raw:
        return ""
    if raw.startswith("/"):
        raise ValueError("output_name must be relative to model/data")
    parts = raw.split("/")
    if any(part in {"", ".", ".."} or safe_run_tag(part) != part for part in parts):
        raise ValueError("output_name contains an unsafe path component")
    return os.path.join(*parts)


def filter_insects_pair(x, y, class_a: int, class_b: int):
    y = y.view(-1).long()
    mask = (y == class_a) | (y == class_b)
    x2 = x[mask]
    y2 = y[mask]
    # relabel: class_a -> 0, class_b -> 1
    ybin = (y2 == class_b).long()
    return x2, ybin


def build_eval_stream(x_S1, y_S1, x_S2, y_S2, B, t):
    """
    Exact same stream segment used by OLD3S_Shallow.FirstPeriod():
      - first B samples from S1
      - first t samples from S2
    """
    y_stream = torch.cat([y_S1[:B], y_S2[:t]], dim=0).view(-1).long()
    return y_stream


def select_contiguous_stream(x_S1, y_S1, x_S2, y_S2, B, t):
    """Select the B rows before and t rows after the temporal boundary."""
    B = int(B)
    t = int(t)
    if B < 0 or t < 0 or B > len(x_S1) or t > len(x_S2):
        raise ValueError("Requested contiguous stream segment is out of bounds")
    s1_x = x_S1[-B:] if B else x_S1[:0]
    s1_y = y_S1[-B:] if B else y_S1[:0]
    return s1_x, s1_y, x_S2[:t], y_S2[:t]


def set_global_min_maj_from_reference(y_reference):
    """
    Fix minority/majority identities from the observed S1 reference period.

    Using the complete evaluated stream would inspect future S2 labels when
    deciding which class is the minority.  That does not affect model fitting,
    but it is avoidable evaluation leakage and can silently change the class
    being reported after the feature transition.
    """
    y_np = y_reference.detach().cpu().numpy().astype(np.int64)
    if y_np.size == 0:
        raise ValueError("The S1 reference period must contain at least one label")
    classes, counts = np.unique(y_np, return_counts=True)

    maj_class = int(classes[np.argmax(counts)])
    min_class = int(classes[np.argmin(counts)])

    evaluator_stream.set_global_min_maj(min_class, maj_class)
    print("[INFO] Fixed S1-reference maj/min:", maj_class, min_class, dict(zip(classes.tolist(), counts.tolist())))

    return min_class, maj_class


def run_global_majority_baseline(y_stream, drift_idx, save_path, metadata=None, eval_window=None):
    """
    Classical baseline:
    predict the full-stream majority class at every step.
    """
    evaluator_stream.reset_stream_metrics(window=eval_window)

    y_np = y_stream.detach().cpu().numpy().astype(np.int64)
    classes, counts = np.unique(y_np, return_counts=True)
    global_majority = int(classes[np.argmax(counts)])

    # assumes labels are encoded to 0..C-1, which matches your current pipeline
    num_classes = int(np.max(y_np)) + 1

    logger = StreamMetricLogger(num_classes=num_classes)
    logger.mark_drift(drift_idx)

    for y_true in y_np:
        logger.start_step()

        proba = np.zeros(num_classes, dtype=np.float32)
        proba[global_majority] = 1.0

        logger.update(y_true=int(y_true), y_proba=proba)
        logger.end_step()

    save_dir = data_path(save_path, "metrics")
    logger.save_npz(save_dir, metadata=metadata)
    logger.stop()

    print(f"[OK] Global majority baseline saved to: {os.path.join(save_dir, 'all_metrics.npz')}")
    print(f"[INFO] Global majority class = {global_majority}")


def run_cumulative_majority_baseline(y_stream, drift_idx, save_path, metadata=None, eval_window=None):
    """
    Prequential cumulative-majority baseline:
    predict from past labels only, then update counts with current label.
    """
    evaluator_stream.reset_stream_metrics(window=eval_window)

    y_np = y_stream.detach().cpu().numpy().astype(np.int64)

    # assumes labels are encoded to 0..C-1, which matches your current pipeline
    num_classes = int(np.max(y_np)) + 1
    default_class = int(np.min(y_np))

    logger = StreamMetricLogger(num_classes=num_classes)
    logger.mark_drift(drift_idx)

    seen_counts = Counter()

    for y_true in y_np:
        logger.start_step()

        if len(seen_counts) == 0:
            pred_class = default_class
        else:
            # tie-break by smaller class id
            pred_class = min(seen_counts.keys(), key=lambda c: (-seen_counts[c], c))

        proba = np.zeros(num_classes, dtype=np.float32)
        proba[pred_class] = 1.0

        logger.update(y_true=int(y_true), y_proba=proba)

        # strictly prequential: update AFTER prediction/evaluation
        seen_counts[int(y_true)] += 1

        logger.end_step()

    save_dir = data_path(save_path, "metrics")
    logger.save_npz(save_dir, metadata=metadata)
    logger.stop()

    print(f"[OK] Cumulative majority baseline saved to: {os.path.join(save_dir, 'all_metrics.npz')}")


def run_no_change_baseline(x_stream, y_stream, drift_idx, save_path, metadata=None, eval_window=None, train_until=None):
    """
    No-change learner:
    train online before the drift boundary, then freeze afterwards.
    """
    evaluator_stream.reset_stream_metrics(window=eval_window)

    x_stream = x_stream.to(DEVICE)
    y_stream = y_stream.view(-1).long().to(DEVICE)

    num_classes = int(torch.max(y_stream).item()) + 1
    in_dim = int(x_stream.shape[1])

    model = MLP(in_planes=in_dim, num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    logger = StreamMetricLogger(num_classes=num_classes)
    logger.mark_drift(drift_idx)

    model.train()
    freeze_after = len(x_stream) if train_until is None else int(train_until)

    for i in range(len(x_stream)):
        x_i = x_stream[i].unsqueeze(0)
        y_i = int(y_stream[i].item())

        logger.start_step()

        logits_list = model(x_i)
        logits = logits_list[-1].squeeze(0)
        proba = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        logger.update(y_true=y_i, y_proba=proba)

        if i < freeze_after:
            loss = criterion(
                logits.unsqueeze(0),
                torch.tensor([y_i], dtype=torch.long, device=DEVICE),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logger.end_step()

    save_dir = data_path(save_path, "metrics")
    logger.save_npz(save_dir, metadata=metadata)
    logger.stop()

    print(f"[OK] No-change baseline saved to: {os.path.join(save_dir, 'all_metrics.npz')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-DataName", type=str, required=True,
                        help="magic, insects")
    parser.add_argument("-AutoEncoder", type=str, default="AE")
    parser.add_argument("-beta", type=float, default=0.9)
    parser.add_argument("-eta", type=float, default=-0.001)
    parser.add_argument("-learningrate", type=float, default=1e-3)
    parser.add_argument(
        "-RecLossFunc",
        type=str,
        default="mse",
        help="Reconstruction loss; MSE is the default for standardized continuous features.",
    )
    parser.add_argument("-T1", type=int, default=5000)
    parser.add_argument("-t", type=int, default=1000)
    parser.add_argument("-eval_window", type=int, default=500)
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("-detector_type", type=str, default="adwin")
    parser.add_argument("-mddm_win", type=int, default=100)
    parser.add_argument("-mddm_ratio", type=float, default=1.01)
    parser.add_argument("-mddm_a_difference", type=float, default=0.01)
    parser.add_argument("-mddm_e_lambda", type=float, default=0.01)
    parser.add_argument("-mddm_delta", type=float, default=1e-6)
    parser.add_argument("-prototype_weight", type=float, default=0.35)
    parser.add_argument("-prototype_bank_size", type=int, default=256)
    parser.add_argument("-prototype_k", type=int, default=5)
    parser.add_argument("-prototype_merge_alpha", type=float, default=0.2)
    parser.add_argument("-prototype_rep_weight", type=float, default=1.0)
    parser.add_argument("-prototype_drift_weight", type=float, default=0.75)
    parser.add_argument("-prototype_minority_weight", type=float, default=0.5)
    parser.add_argument("-prototype_uncertainty_weight", type=float, default=0.35)
    parser.add_argument("-prototype_obsolescence_weight", type=float, default=1.0)
    parser.add_argument("-prototype_freshness_weight", type=float, default=1.0)
    parser.add_argument("-pset_max", type=int, default=128)
    parser.add_argument("-pset_drift_k", type=int, default=8)
    parser.add_argument("-pset_drift_threshold", type=int, default=4)
    parser.add_argument("-use_transfer_mapper", type=int, choices=[0, 1], default=1)
    parser.add_argument("-transfer_mapper_mode", choices=["residual", "mlp"], default="residual")
    parser.add_argument("-use_historical_knowledge", type=int, choices=[0, 1], default=1)
    parser.add_argument("-use_prototype_memory", type=int, choices=[0, 1], default=1)
    parser.add_argument("-fusion_mode", choices=["moe", "fixed"], default="moe")
    parser.add_argument("-enable_historical_expert", type=int, choices=[0, 1], default=1)
    parser.add_argument("-enable_adaptive_expert", type=int, choices=[0, 1], default=1)
    parser.add_argument("-enable_prototype_expert", type=int, choices=[0, 1], default=1)
    parser.add_argument("-router_hidden_dim", type=int, default=32)
    parser.add_argument("-forgetting_reference_size", type=int, default=128)
    parser.add_argument("-diagnostic_interval", type=int, default=10)
    parser.add_argument("-run_tag", type=str, default="")
    parser.add_argument(
        "-output_name",
        type=str,
        default="",
        help="Optional safe directory name under model/data for orchestrated runs.",
    )
    parser.add_argument("-run_sanity_baselines", type=int, choices=[0, 1], default=1)
    parser.add_argument("-protocol_name", type=str, default="old3s_feature_evolution")
    parser.add_argument("-feature_protocol", type=str, default="feature_evolution")
    parser.add_argument("-shared_frac", type=float, default=0.5)
    parser.add_argument("-split_ratio", type=float, default=0.8)
    parser.add_argument(
        "-split_index",
        type=int,
        default=-1,
        help="Exact original-row S1/S2 boundary; overrides split_ratio when non-negative.",
    )
    parser.add_argument("-feature_seed", type=int, default=1314)
    parser.add_argument(
        "-feature_scenario",
        choices=["balanced", "s2_expands", "s2_contracts"],
        default="balanced",
        help="How non-shared features are divided between S1 and S2.",
    )
    parser.add_argument(
        "-insects_csv",
        type=str,
        default=data_path("INSECTS_incremental_imbalanced.csv"),
    )
    parser.add_argument("-class_a", type=int, default=None, help="INSECTS class id A for one-vs-one binary")
    parser.add_argument("-class_b", type=int, default=None, help="INSECTS class id B for one-vs-one binary")

    args = parser.parse_args()

    dataname = args.DataName.strip().lower()
    detector_type = str(args.detector_type).strip().lower()
    class_pair = None
    set_global_seed(args.seed)

    if dataname == "magic":
        x_S1, y_S1, x_S2, y_S2 = loadmagic()
        path = "parameter_magic"
        feature_metadata = {
            "feature_protocol": "random_projection",
            "feature_scenario": "s2_expands",
            "dimension1": int(x_S1.shape[1]),
            "dimension2": int(x_S2.shape[1]),
        }
    elif dataname == "insects":
        x_S1, y_S1, x_S2, y_S2, feature_metadata = loadinsects(
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
        csv_stem = os.path.splitext(os.path.basename(args.insects_csv))[0]
        proto_tag = str(args.feature_protocol).strip().lower()
        path = f"parameter_insects__{csv_stem}__{proto_tag}"
        if args.feature_scenario != "balanced":
            path = f"{path}__{args.feature_scenario}"
    else:
        raise ValueError(f"Unsupported DataName: {args.DataName}")

    if detector_type != "adwin":
        path = f"{path}__{detector_type}"
    run_tag = safe_run_tag(args.run_tag)
    if run_tag:
        path = f"{path}__{run_tag}"

    # IMPORTANT: dimension1 is S1 feature dim, dimension2 is S2 feature dim
    dimension1 = int(x_S1.shape[1])
    dimension2 = int(x_S2.shape[1])

    # make sure labels are long
    y_S1 = y_S1.view(-1).long()
    y_S2 = y_S2.view(-1).long()

    if dataname == "insects" and args.class_a is not None and args.class_b is not None:
        a = int(args.class_a)
        b = int(args.class_b)
        class_pair = f"{a}vs{b}"

        x_S1, y_S1 = filter_insects_pair(x_S1, y_S1, a, b)
        x_S2, y_S2 = filter_insects_pair(x_S2, y_S2, a, b)

        print(f"[INFO] INSECTS one-vs-one enabled: {a} vs {b}")
        print("[INFO] S1 counts:", torch.bincount(y_S1).tolist())
        print("[INFO] S2 counts:", torch.bincount(y_S2).tolist())

        # hard safety check
        assert int(torch.unique(torch.cat([y_S1, y_S2])).numel()) == 2

        # unique output folder per pair (prevents overwriting results)
        csv_stem = os.path.splitext(os.path.basename(args.insects_csv))[0]
        proto_tag = str(args.feature_protocol).strip().lower()
        path = f"parameter_insects__{csv_stem}__{proto_tag}__pair_{a}vs{b}__{detector_type}"
        if args.feature_scenario != "balanced":
            path = f"{path}__{args.feature_scenario}"
        if run_tag:
            path = f"{path}__{run_tag}"
        print(f"[INFO] Output path set to: {path}")

    output_name = safe_output_name(args.output_name)
    if output_name:
        path = output_name

    # Choose T1/t safely relative to available data:
    # S1 available = len(x_S1), S2 available = len(x_S2)
    # We run B = T1 - t from S1, then t from S2.
    # So must have B <= len(S1) and t <= len(S2).
    t = min(args.t, len(x_S2))
    B = min(args.T1 - t, len(x_S1))
    T1 = B + t

    # Preserve stream continuity for temporal datasets. The evaluated S1 segment
    # must end immediately before the S1/S2 boundary; taking x_S1[:B] would skip
    # all observations between B and the split index.
    if dataname == "insects":
        split_idx = int(feature_metadata["split_index"])
        if class_pair is None:
            stream_start_original = split_idx - B
            stream_end_original = split_idx + t
            stream_coordinate_system = "original_row"
        else:
            # Filtering preserves order but removes rows, so offsets are in the
            # filtered subsequence and cannot be claimed as original row IDs.
            stream_start_original = -1
            stream_end_original = -1
            stream_coordinate_system = "class_filtered_subsequence"
        x_S1, y_S1, x_S2, y_S2 = select_contiguous_stream(
            x_S1, y_S1, x_S2, y_S2, B, t
        )
    else:
        stream_start_original = 0
        stream_end_original = T1
        stream_coordinate_system = "dataset_sequence"

    print(f"[INFO] Using T1={T1} (B={B} from S1, t={t} from S2)")
    print(f"[INFO] dims: S1={dimension1}, S2={dimension2}")
    print(f"[INFO] eval_window={args.eval_window}, detector={detector_type}, seed={args.seed}")
    print(f"[INFO] protocol={args.protocol_name}")
    if dataname == "insects":
        if class_pair is None:
            print(
                "[INFO] contiguous original stream interval: "
                f"[{stream_start_original}, {stream_end_original}), "
                f"transition={feature_metadata['split_index']}"
            )
        else:
            print("[INFO] stream is an order-preserving class-filtered subsequence")
    if dataname == "insects":
        if str(args.feature_protocol).strip().lower() == "feature_evolution":
            print("[INFO] INSECTS uses OLD3S-style feature evolution: S1=obsolete+shared, S2=shared+new.")
            print(
                f"[INFO] shared_frac={args.shared_frac}, feature_seed={args.feature_seed}, "
                f"scenario={args.feature_scenario}"
            )
            print(
                "[INFO] feature counts: "
                f"old_only={feature_metadata['old_only_count']}, "
                f"shared={feature_metadata['shared_count']}, "
                f"new_only={feature_metadata['new_only_count']}"
            )
        else:
            print("[WARN] INSECTS is using a same-feature stream split, not true feature evolution/obsolescence.")

    run_metadata = {
        "dataset": dataname,
        "run_path": path,
        "protocol_name": str(args.protocol_name),
        "protocol_revision": PROTOCOL_REVISION,
        "autoencoder": args.AutoEncoder,
        "beta": float(args.beta),
        "eta": float(args.eta),
        "learningrate": float(args.learningrate),
        "rec_loss": str(args.RecLossFunc),
        "detector_type": detector_type,
        "mddm_win": int(args.mddm_win),
        "mddm_ratio": float(args.mddm_ratio),
        "mddm_a_difference": float(args.mddm_a_difference),
        "mddm_e_lambda": float(args.mddm_e_lambda),
        "mddm_delta": float(args.mddm_delta),
        "T1": int(T1),
        "B": int(B),
        "t": int(t),
        "dimension1": int(dimension1),
        "dimension2": int(dimension2),
        "window_size": int(args.eval_window),
        "insects_csv": str(args.insects_csv) if dataname == "insects" else "",
        "feature_protocol": str(args.feature_protocol),
        "shared_frac": float(args.shared_frac),
        "feature_seed": int(args.feature_seed),
        "feature_scenario": str(args.feature_scenario),
        "feature_metadata": feature_metadata,
        "split_ratio": float(feature_metadata.get("split_ratio", args.split_ratio)),
        "split_index_requested": int(args.split_index),
        "stream_start_original": int(stream_start_original),
        "transition_original": int(feature_metadata.get("split_index", B)),
        "stream_end_original": int(stream_end_original),
        "stream_coordinate_system": stream_coordinate_system,
        "prototype_weight": float(args.prototype_weight),
        "prototype_bank_size": int(args.prototype_bank_size),
        "prototype_k": int(args.prototype_k),
        "prototype_merge_alpha": float(args.prototype_merge_alpha),
        "prototype_rep_weight": float(args.prototype_rep_weight),
        "prototype_drift_weight": float(args.prototype_drift_weight),
        "prototype_minority_weight": float(args.prototype_minority_weight),
        "prototype_uncertainty_weight": float(args.prototype_uncertainty_weight),
        "prototype_obsolescence_weight": float(args.prototype_obsolescence_weight),
        "prototype_freshness_weight": float(args.prototype_freshness_weight),
        "pset_max": int(args.pset_max),
        "pset_drift_k": int(args.pset_drift_k),
        "pset_drift_threshold": int(args.pset_drift_threshold),
        "use_transfer_mapper": bool(args.use_transfer_mapper),
        "transfer_mapper_mode": str(args.transfer_mapper_mode),
        "use_historical_knowledge": bool(args.use_historical_knowledge),
        "use_prototype_memory": bool(args.use_prototype_memory),
        "fusion_mode": str(args.fusion_mode),
        "enable_historical_expert_requested": bool(args.enable_historical_expert),
        "enable_historical_expert_effective": bool(args.enable_historical_expert and args.use_historical_knowledge),
        "enable_adaptive_expert": bool(args.enable_adaptive_expert),
        "enable_prototype_expert_requested": bool(args.enable_prototype_expert),
        "enable_prototype_expert_effective": bool(args.enable_prototype_expert and args.use_prototype_memory),
        "router_hidden_dim": int(args.router_hidden_dim),
        "forgetting_reference_size": int(args.forgetting_reference_size),
        "diagnostic_interval": int(args.diagnostic_interval),
        "run_tag": run_tag,
        "output_name": output_name,
        "class_pair": class_pair or "",
        "seed": int(args.seed),
    }
    stream_annotation = (
        get_stream_annotation(args.insects_csv) if dataname == "insects" else None
    )
    if stream_annotation is not None and class_pair is None:
        abrupt_original = stream_annotation.get("exact_abrupt_points", [])
        reference_original = stream_annotation.get("reference_points", [])
        abrupt_local = [
            int(point - stream_start_original)
            for point in abrupt_original
            if stream_start_original <= point < stream_end_original
        ]
        reference_local = [
            int(point - stream_start_original)
            for point in reference_original
            if stream_start_original <= point < stream_end_original
        ]
    else:
        abrupt_local = []
        reference_local = []
    run_metadata.update(
        {
            "stream_annotation": stream_annotation or {},
            "known_abrupt_points_local": abrupt_local,
            "reference_change_points_local": reference_local,
            "feature_transition_local": int(B),
        }
    )
    if stream_annotation is not None:
        print(
            f"[INFO] known change pattern={stream_annotation['change_pattern']}, "
            f"local exact abrupt points={abrupt_local}, "
            f"local reference points={reference_local}"
        )

    # Build the exact evaluated stream for all runs
    y_stream = build_eval_stream(x_S1, y_S1, x_S2, y_S2, B, t)

    # Fix reported minority/majority identities using only labels already seen
    # before S2 begins. This avoids future-label leakage in metric definitions.
    min_class, maj_class = set_global_min_maj_from_reference(y_S1[:B])
    run_metadata.update(
        {
            "minority_class": int(min_class),
            "majority_class": int(maj_class),
            "minority_reference": "evaluated_s1_segment_only",
        }
    )

    # Reset rolling evaluator before adaptive run
    evaluator_stream.reset_stream_metrics(window=args.eval_window)

    # -------------------------
    # Adaptive OLD3S run
    # -------------------------
    model = OLD3S_Shallow(
        x_S1, y_S1,
        x_S2, y_S2,
        T1=T1, t=t,
        dimension1=dimension1, dimension2=dimension2,
        path=path,
        lr=args.learningrate,
        b=args.beta,
        eta=args.eta,
        RecLossFunc=args.RecLossFunc,
        detector_type=detector_type,
        mddm_win=args.mddm_win,
        mddm_ratio=args.mddm_ratio,
        mddm_a_difference=args.mddm_a_difference,
        mddm_e_lambda=args.mddm_e_lambda,
        mddm_delta=args.mddm_delta,
        prototype_weight=args.prototype_weight,
        prototype_bank_size=args.prototype_bank_size,
        prototype_k=args.prototype_k,
        prototype_merge_alpha=args.prototype_merge_alpha,
        prototype_rep_weight=args.prototype_rep_weight,
        prototype_drift_weight=args.prototype_drift_weight,
        prototype_minority_weight=args.prototype_minority_weight,
        prototype_uncertainty_weight=args.prototype_uncertainty_weight,
        prototype_obsolescence_weight=args.prototype_obsolescence_weight,
        prototype_freshness_weight=args.prototype_freshness_weight,
        pset_max=args.pset_max,
        pset_drift_k=args.pset_drift_k,
        pset_drift_threshold=args.pset_drift_threshold,
        use_transfer_mapper=bool(args.use_transfer_mapper),
        transfer_mapper_mode=args.transfer_mapper_mode,
        use_historical_knowledge=bool(args.use_historical_knowledge),
        use_prototype_memory=bool(args.use_prototype_memory),
        fusion_mode=args.fusion_mode,
        enable_historical_expert=bool(args.enable_historical_expert),
        enable_adaptive_expert=bool(args.enable_adaptive_expert),
        enable_prototype_expert=bool(args.enable_prototype_expert),
        router_hidden_dim=args.router_hidden_dim,
        forgetting_reference_size=args.forgetting_reference_size,
        diagnostic_interval=args.diagnostic_interval,
        run_metadata={**run_metadata, "method": "adaptive"},
    )

    model.FirstPeriod()
    print("[OK] Adaptive run done.")

    if not args.run_sanity_baselines:
        print("[INFO] Sanity baselines disabled for this orchestrated run.")
        print("[OK] Done.")
        return

    # -------------------------
    # No-change baseline
    # -------------------------
    if dimension1 == dimension2:
        x_stream = torch.cat([x_S1[:B], x_S2[:t]], dim=0)
        run_no_change_baseline(
            x_stream=x_stream,
            y_stream=y_stream,
            drift_idx=B,
            save_path=f"{path}__nochange",
            metadata={**run_metadata, "method": "no_change"},
            eval_window=args.eval_window,
            train_until=B,
        )
    else:
        print("[WARN] Skipping no-change baseline because S1 and S2 feature dimensions differ.")

    # -------------------------
    # Global majority baseline
    # -------------------------
    run_global_majority_baseline(
        y_stream=y_stream,
        drift_idx=B,
        save_path=f"{path}__global_majority",
        metadata={**run_metadata, "method": "global_majority"},
        eval_window=args.eval_window,
    )

    # -------------------------
    # Cumulative majority baseline
    # -------------------------
    run_cumulative_majority_baseline(
        y_stream=y_stream,
        drift_idx=B,
        save_path=f"{path}__cumulative_majority",
        metadata={**run_metadata, "method": "cumulative_majority"},
        eval_window=args.eval_window,
    )

    print("[OK] Done.")


if __name__ == "__main__":
    main()
