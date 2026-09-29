# train.py
import argparse
import os
import random
import re
import torch
import numpy as np

import evaluator_stream
from loaddatasets import transition_location_info, prepare_training_stream
from model import OLD3S_Shallow
from paths import data_path


PROTOCOL_REVISION = "thesis_protocol_v7"


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


def training_output_name(args, detector_type, run_tag):
    """Build the default name once; preserve names of existing experiment variants."""
    dataset = args.DataName.strip().lower()
    if dataset == "insects":
        stem = os.path.splitext(os.path.basename(args.insects_csv))[0]
        parts = ["parameter_insects", stem, args.feature_protocol.strip().lower()]
        if args.feature_scenario != "balanced":
            parts.append(args.feature_scenario)
    else:
        canonical = "new-thyroid" if dataset in {"thyroid", "new_thyroid"} else dataset
        parts = [f"parameter_{canonical.replace('-', '_')}"]
    if detector_type != "adwin":
        parts.append(detector_type)
    if run_tag:
        parts.append(run_tag)
    return "__".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-DataName", type=str, required=True,
                        help="magic, adult, arrhythmia, car, new-thyroid, insects")
    parser.add_argument("-AutoEncoder", type=str, default="AE")
    parser.add_argument("-beta", type=float, default=0.9)
    parser.add_argument("-eta", type=float, default=-0.001)
    parser.add_argument("-learningrate", type=float, default=1e-3)
    parser.add_argument("-alignment_weight", type=float, default=0.2)
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
    parser.add_argument("-use_historical_knowledge", type=int, choices=[0, 1], default=1)
    parser.add_argument("-use_prototype_memory", type=int, choices=[0, 1], default=1)
    parser.add_argument("-fusion_mode", choices=["moe", "fixed"], default="moe")
    parser.add_argument("-enable_adaptive_expert", type=int, choices=[0, 1], default=1)
    parser.add_argument("-enable_historical_expert", type=int, choices=[0, 1], default=1)
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
        "-static_calibration_size",
        type=int,
        default=500,
        help="Paired independent-dataset rows reserved before evaluated S1/S2.",
    )
    parser.add_argument("-protocol_revision", type=str, default=PROTOCOL_REVISION)
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

    parser.add_argument("-transition_location", default="")
    parser.add_argument("-transition_fraction", type=float, default=None)
    parser.add_argument("-evaluation_region_start", type=int, default=None)
    parser.add_argument("-evaluation_region_end", type=int, default=None)

    args = parser.parse_args()

    dataname = args.DataName.strip().lower()
    detector_type = str(args.detector_type).strip().lower()
    set_global_seed(args.seed)
    (x_S1, y_S1, x_S2, y_S2), stream_info = prepare_training_stream(args)
    feature_metadata = stream_info["feature_metadata"]
    B, t = len(x_S1), len(x_S2)
    T1 = B + t
    dimension1, dimension2 = int(x_S1.shape[1]), int(x_S2.shape[1])
    run_tag = safe_run_tag(args.run_tag)
    output_name = safe_output_name(args.output_name)
    path = output_name or training_output_name(args, detector_type, run_tag)

    print(f"[INFO] Using T1={T1} (B={B} from S1, t={t} from S2)")
    print(f"[INFO] dims: S1={dimension1}, S2={dimension2}")
    print(f"[INFO] eval_window={args.eval_window}, detector={detector_type}, seed={args.seed}")
    print(f"[INFO] stream coordinates={stream_info['stream_coordinate_system']}, "
          f"interval=[{stream_info['stream_start_original']}, {stream_info['stream_end_original']}), "
          f"transition={stream_info['transition_original']}")

    run_info = {
        "dataset": dataname,
        "run_path": path,
        "protocol_name": str(args.protocol_name),
        "protocol_revision": str(args.protocol_revision),
        "autoencoder": args.AutoEncoder,
        "beta": float(args.beta),
        "eta": float(args.eta),
        "learningrate": float(args.learningrate),
        "alignment_weight": float(args.alignment_weight),
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
        "split_ratio": float(feature_metadata.get("split_ratio", args.split_ratio)),
        "static_calibration_size": int(args.static_calibration_size) if dataname != "insects" else 0,
        "split_index_requested": int(args.split_index),
        **stream_info,
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
        "use_historical_knowledge": bool(args.use_historical_knowledge),
        "use_prototype_memory": bool(args.use_prototype_memory),
        "fusion_mode": str(args.fusion_mode),
        "enable_adaptive_expert": bool(args.enable_adaptive_expert),
        "enable_historical_expert": bool(args.enable_historical_expert),
        "enable_prototype_expert": bool(args.enable_prototype_expert),
        "router_hidden_dim": int(args.router_hidden_dim),
        "forgetting_reference_size": int(args.forgetting_reference_size),
        "diagnostic_interval": int(args.diagnostic_interval),
        "run_tag": run_tag,
        "output_name": output_name,
        "seed": int(args.seed),
    }
    run_info.update(transition_location_info(
        args, stream_info["stream_start_original"], stream_info["stream_end_original"], stream_info["transition_original"],
        feature_metadata.get("total_instances", len(x_S1) + len(x_S2))
    ))

    # Fix reported minority/majority identities using only labels already seen
    # before S2 begins. This avoids future-label leakage in metric definitions.
    min_class, maj_class = set_global_min_maj_from_reference(y_S1[:B])
    run_info.update(
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
        alignment_weight=args.alignment_weight,
        b=args.beta,
        eta=args.eta,
        RecLossFunc=args.RecLossFunc,
        detector_options={
            'detector_type': detector_type,
            'mddm_win': args.mddm_win,
            'mddm_ratio': args.mddm_ratio,
            'mddm_a_difference': args.mddm_a_difference,
            'mddm_e_lambda': args.mddm_e_lambda,
            'mddm_delta': args.mddm_delta,
        },
        prototype_options={
            'weight': args.prototype_weight,
            'bank_size': args.prototype_bank_size,
            'k': args.prototype_k,
            'merge_alpha': args.prototype_merge_alpha,
            'rep_weight': args.prototype_rep_weight,
            'drift_weight': args.prototype_drift_weight,
            'minority_weight': args.prototype_minority_weight,
            'uncertainty_weight': args.prototype_uncertainty_weight,
            'obsolescence_weight': args.prototype_obsolescence_weight,
            'freshness_weight': args.prototype_freshness_weight,
            'pset_max': args.pset_max,
            'pset_drift_k': args.pset_drift_k,
            'pset_drift_threshold': args.pset_drift_threshold,
        },
        use_transfer_mapper=bool(args.use_transfer_mapper),
        use_historical_knowledge=bool(args.use_historical_knowledge),
        use_prototype_memory=bool(args.use_prototype_memory),
        fusion_mode=args.fusion_mode,
        enable_adaptive_expert=bool(args.enable_adaptive_expert),
        enable_historical_expert=bool(args.enable_historical_expert),
        enable_prototype_expert=bool(args.enable_prototype_expert),
        router_hidden_dim=args.router_hidden_dim,
        forgetting_reference_size=args.forgetting_reference_size,
        diagnostic_interval=args.diagnostic_interval,
        run_info={**run_info, "method": "adaptive"},
    )

    model.FirstPeriod()
    print("[OK] Adaptive run done.")


if __name__ == "__main__":
    main()
