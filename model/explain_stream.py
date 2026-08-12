"""Phase-faithful SHAP and LIME analysis for a saved thesis stream run."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

_MPL_CACHE = Path(__file__).resolve().parent / "data" / ".matplotlib_cache"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import shap
import torch
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

from autoencoder import AutoEncoder_Shallow
from loaddatasets import loadinsects
from mlp import MLP
from model import OLD3S_Shallow
from moe import FeatureDriftRouter, MoEFusion
from paths import MODEL_DIR
from train import select_contiguous_stream


def write_csv(rows: Iterable[dict], path: Path) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_dataset(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (Path(MODEL_DIR) / path).resolve()


def load_stream(metadata: dict):
    feature = metadata["feature_metadata"]
    x1, y1, x2, y2, loaded_feature = loadinsects(
        str(resolve_dataset(metadata["insects_csv"])),
        split_ratio=float(metadata.get("split_ratio", 0.8)),
        split_index=int(feature["split_index"]),
        feature_protocol=metadata["feature_protocol"],
        shared_frac=float(metadata["shared_frac"]),
        feature_seed=int(metadata["feature_seed"]),
        feature_scenario=metadata["feature_scenario"],
        scaler_exclusion_size=int(metadata["B"]),
        return_metadata=True,
    )
    x1, y1, x2, y2 = select_contiguous_stream(
        x1, y1, x2, y2, int(metadata["B"]), int(metadata["t"])
    )
    if loaded_feature["s1_indices"] != feature["s1_indices"] or loaded_feature["s2_indices"] != feature["s2_indices"]:
        raise ValueError("Reconstructed feature partition does not match checkpoint metadata")
    return x1, y1, x2, y2


class SnapshotPredictor:
    """Reconstruct the complete fused predictor at one online snapshot."""

    def __init__(self, checkpoint: dict, x1, y1, x2, y2):
        metadata = checkpoint["run_metadata"]
        self.metadata = metadata
        self.device = torch.device("cpu")
        self.wrapper = OLD3S_Shallow(
            x1,
            y1,
            x2,
            y2,
            T1=int(metadata["T1"]),
            t=int(metadata["t"]),
            dimension1=int(checkpoint["dimension1"]),
            dimension2=int(checkpoint["dimension2"]),
            path="explainability_reconstruction_only",
            lr=float(metadata["learningrate"]),
            b=float(metadata["beta"]),
            eta=float(metadata["eta"]),
            RecLossFunc=metadata["rec_loss"],
            prototype_weight=float(metadata["prototype_weight"]),
            prototype_bank_size=int(metadata["prototype_bank_size"]),
            prototype_k=int(metadata["prototype_k"]),
            prototype_merge_alpha=float(metadata["prototype_merge_alpha"]),
            prototype_rep_weight=float(metadata["prototype_rep_weight"]),
            prototype_drift_weight=float(metadata["prototype_drift_weight"]),
            prototype_minority_weight=float(metadata["prototype_minority_weight"]),
            prototype_uncertainty_weight=float(metadata["prototype_uncertainty_weight"]),
            prototype_obsolescence_weight=float(metadata["prototype_obsolescence_weight"]),
            prototype_freshness_weight=float(metadata["prototype_freshness_weight"]),
            use_transfer_mapper=bool(metadata["use_transfer_mapper"]),
            transfer_mapper_mode=metadata.get("transfer_mapper_mode", checkpoint.get("transfer_mapper_mode", "mlp")),
            use_historical_knowledge=bool(metadata["use_historical_knowledge"]),
            use_prototype_memory=bool(metadata["use_prototype_memory"]),
            fusion_mode=metadata["fusion_mode"],
            enable_historical_expert=bool(metadata["enable_historical_expert_requested"]),
            enable_adaptive_expert=bool(metadata["enable_adaptive_expert"]),
            enable_prototype_expert=bool(metadata["enable_prototype_expert_requested"]),
            router_hidden_dim=int(checkpoint["router_hidden_dim"]),
            run_metadata=metadata,
        )
        self.wrapper.autoencoder_1.load_state_dict(checkpoint["autoencoder_1"])
        self.wrapper.autoencoder_2.load_state_dict(checkpoint["autoencoder_2"])
        self.wrapper.transfer_mapper.load_state_dict(checkpoint["transfer_mapper"])
        self.classifier_1 = MLP(checkpoint["dimension2"], checkpoint["num_classes"])
        self.classifier_1.load_state_dict(checkpoint["classifier_1"])
        self.classifier_2 = None
        if checkpoint["classifier_2"] is not None:
            self.classifier_2 = MLP(checkpoint["dimension2"], checkpoint["num_classes"])
            self.classifier_2.load_state_dict(checkpoint["classifier_2"])
        self.moe = None
        if checkpoint["moe_fusion"] is not None:
            router = FeatureDriftRouter(
                int(checkpoint["dimension2"]) + 2,
                int(checkpoint["router_hidden_dim"]),
            )
            self.moe = MoEFusion(router)
            self.moe.load_state_dict(checkpoint["moe_fusion"])
        self.wrapper.alpha = torch.nn.Parameter(checkpoint["alpha_s1_final"].clone(), requires_grad=False)
        self.wrapper.alpha_historical = (
            torch.nn.Parameter(checkpoint["alpha_historical"].clone(), requires_grad=False)
            if checkpoint["alpha_historical"] is not None
            else None
        )
        self.wrapper.alpha_adaptive = (
            torch.nn.Parameter(checkpoint["alpha_adaptive"].clone(), requires_grad=False)
            if checkpoint["alpha_adaptive"] is not None
            else None
        )
        self.wrapper.prototype_bank = [
            {**prototype, "vec": prototype["vec"].clone()}
            for prototype in checkpoint["prototype_bank"]
        ]
        self.wrapper.prototype_step = int(checkpoint.get("prototype_step", checkpoint["snapshot_step"] + 1))
        self.wrapper.class_proto_sum.copy_(checkpoint["class_proto_sum"])
        self.wrapper.class_proto_count.copy_(checkpoint["class_proto_count"])
        for module in (
            self.wrapper.autoencoder_1,
            self.wrapper.autoencoder_2,
            self.wrapper.transfer_mapper,
            self.classifier_1,
            self.classifier_2,
            self.moe,
        ):
            if module is not None:
                module.eval()

    @torch.no_grad()
    def predict_proba(self, values: np.ndarray, phase: str) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        outputs = []
        for row in values:
            x = torch.from_numpy(row).view(1, -1)
            if phase == "s1":
                z, _ = self.wrapper.autoencoder_1(x)
                logits = self.wrapper._hb_logits(self.classifier_1, z, self.wrapper.alpha)
                logits = self.wrapper._combined_logits(logits, z, phase="s1")
            else:
                if self.classifier_2 is None or self.moe is None:
                    raise ValueError("S2 explanation requested from a pre-S2 snapshot")
                z, _ = self.wrapper.autoencoder_2(x)
                z_historical = self.wrapper._map_to_historical(z)
                historical = (
                    self.wrapper._hb_logits(self.classifier_1, z_historical, self.wrapper.alpha_historical)
                    if self.wrapper.enable_historical_expert
                    else torch.zeros((1, self.wrapper.num_classes))
                )
                adaptive = (
                    self.wrapper._hb_logits(self.classifier_2, z, self.wrapper.alpha_adaptive)
                    if self.wrapper.enable_adaptive_expert
                    else torch.zeros((1, self.wrapper.num_classes))
                )
                logits, _ = self.wrapper._moe_logits(
                    self.moe,
                    z,
                    historical,
                    adaptive,
                    phase="s2",
                    prototype_z=z_historical,
                )
            outputs.append(torch.softmax(logits, dim=-1).squeeze(0).numpy())
        return np.asarray(outputs, dtype=np.float64)


def phase_for_snapshot(checkpoint: dict) -> str:
    return "s1" if int(checkpoint["snapshot_step"]) < int(checkpoint["run_metadata"]["B"]) else "s2"


def sample_pool(checkpoint: dict, x1, x2, count: int) -> Tuple[np.ndarray, np.ndarray, str]:
    step = int(checkpoint["snapshot_step"])
    boundary = int(checkpoint["run_metadata"]["B"])
    phase = phase_for_snapshot(checkpoint)
    if phase == "s1":
        end = min(len(x1), step + 1)
        start = max(0, end - count)
        return x1[start:end].numpy(), np.arange(start, end), phase
    local_end = min(len(x2), step - boundary + 1)
    local_start = max(0, local_end - count)
    return x2[local_start:local_end].numpy(), boundary + np.arange(local_start, local_end), phase


def normalize_shap_values(values: np.ndarray, n_samples: int, n_features: int, n_classes: int) -> np.ndarray:
    values = np.asarray(values)
    if values.shape == (n_samples, n_features, n_classes):
        return values
    if values.shape == (n_classes, n_samples, n_features):
        return np.moveaxis(values, 0, -1)
    if values.shape == (n_samples, n_features) and n_classes == 1:
        return values[:, :, None]
    raise ValueError(f"Unexpected SHAP value shape {values.shape}")


def top_k_set(weights: np.ndarray, k: int) -> set:
    k = min(max(1, int(k)), len(weights))
    return set(np.argsort(-np.abs(weights))[:k].tolist())


def rank_agreement(first: np.ndarray, second: np.ndarray, k: int) -> Tuple[float, float]:
    first_abs = np.abs(first)
    second_abs = np.abs(second)
    correlation = (
        float("nan")
        if np.ptp(first_abs) <= 1e-15 or np.ptp(second_abs) <= 1e-15
        else spearmanr(first_abs, second_abs).statistic
    )
    first_top, second_top = top_k_set(first, k), top_k_set(second, k)
    jaccard = len(first_top & second_top) / max(1, len(first_top | second_top))
    return float(correlation), float(jaccard)


def representative_indices(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    minority_class: int,
    count: int,
) -> Tuple[np.ndarray, List[str]]:
    """Deterministically cover correct, error, minority, and uncertain cases."""
    predictions = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    selected: List[int] = []
    reasons: Dict[int, set] = {}

    def add(index: Optional[int], reason: str) -> None:
        if index is None:
            return
        index = int(index)
        reasons.setdefault(index, set()).add(reason)
        if index not in selected and len(selected) < count:
            selected.append(index)

    correct = np.flatnonzero(predictions == y_true)
    incorrect = np.flatnonzero(predictions != y_true)
    minority = np.flatnonzero(y_true == int(minority_class))
    add(int(correct[0]) if correct.size else None, "correct")
    add(int(incorrect[0]) if incorrect.size else None, "incorrect")
    add(int(minority[0]) if minority.size else None, "minority")
    add(int(np.argmin(confidence)) if confidence.size else None, "low_confidence")
    for index in np.linspace(0, max(0, len(y_true) - 1), min(count, len(y_true))).astype(int):
        add(int(index), "temporal_coverage")
    for index in range(len(y_true)):
        add(index, "fill")
        if len(selected) >= count:
            break
    return np.asarray(selected, dtype=int), ["|".join(sorted(reasons[index])) for index in selected]


def analyze_snapshot(
    checkpoint_path: Path,
    x1,
    y1,
    x2,
    y2,
    output_dir: Path,
    background_size: int,
    explain_size: int,
    shap_nsamples: int,
    lime_num_samples: int,
    lime_repeats: int,
    top_k: int,
    seed: int,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    predictor = SnapshotPredictor(checkpoint, x1, y1, x2, y2)
    phase = phase_for_snapshot(checkpoint)
    rng = np.random.default_rng(seed)
    background_pool, _, _ = sample_pool(checkpoint, x1, x2, max(background_size * 4, background_size))
    if len(background_pool) == 0:
        raise ValueError(f"No background observations for {checkpoint_path}")
    background_indices = np.linspace(0, len(background_pool) - 1, min(background_size, len(background_pool))).astype(int)
    background = background_pool[background_indices]
    feature_indices = predictor.metadata["feature_metadata"][f"{phase}_indices"]
    feature_names = [f"feature_{index}" for index in feature_indices]
    num_classes = int(checkpoint["num_classes"])
    predict = lambda values: predictor.predict_proba(values, phase)
    explain_pool, stream_indices, _ = sample_pool(
        checkpoint, x1, x2, max(explain_size * 10, 50)
    )
    if phase == "s1":
        candidate_labels = y1[torch.as_tensor(stream_indices, dtype=torch.long)].numpy()
    else:
        local_indices = torch.as_tensor(
            stream_indices - int(predictor.metadata["B"]), dtype=torch.long
        )
        candidate_labels = y2[local_indices].numpy()
    candidate_probabilities = predict(explain_pool)
    selected, selection_reasons = representative_indices(
        candidate_labels,
        candidate_probabilities,
        int(predictor.metadata["minority_class"]),
        min(explain_size, len(explain_pool)),
    )
    explained = explain_pool[selected]
    explained_indices = stream_indices[selected]
    true_labels = candidate_labels[selected].astype(np.int64)

    start = time.perf_counter()
    kernel = shap.KernelExplainer(predict, background, feature_names=feature_names)
    shap_raw = kernel.shap_values(explained, nsamples=int(shap_nsamples), silent=True)
    shap_seconds = time.perf_counter() - start
    shap_values = normalize_shap_values(
        shap_raw, len(explained), len(feature_names), num_classes
    )
    probabilities = predict(explained)
    predictions = np.argmax(probabilities, axis=1)

    global_rows = []
    for feature_position, feature_name in enumerate(feature_names):
        global_rows.append(
            {
                "snapshot": checkpoint_path.stem,
                "phase": phase,
                "feature": feature_name,
                "original_feature_index": int(feature_indices[feature_position]),
                "mean_abs_shap_all_classes": float(np.mean(np.abs(shap_values[:, feature_position, :]))),
                **{
                    f"mean_abs_shap_class_{class_index}": float(
                        np.mean(np.abs(shap_values[:, feature_position, class_index]))
                    )
                    for class_index in range(num_classes)
                },
            }
        )

    lime_rows: List[dict] = []
    agreement_rows: List[dict] = []
    faithfulness_rows: List[dict] = []
    lime_start = time.perf_counter()
    background_mean = np.mean(background, axis=0)
    for sample_position, (sample, stream_index, predicted_class) in enumerate(
        zip(explained, explained_indices, predictions)
    ):
        repeat_weights = []
        for repeat in range(lime_repeats):
            lime_explainer = LimeTabularExplainer(
                background,
                feature_names=feature_names,
                class_names=[f"class_{index}" for index in range(num_classes)],
                mode="classification",
                discretize_continuous=False,
                random_state=seed + repeat,
            )
            explanation = lime_explainer.explain_instance(
                sample,
                predict,
                labels=[int(predicted_class)],
                num_features=len(feature_names),
                num_samples=int(lime_num_samples),
                model_regressor=Ridge(alpha=1.0),
            )
            weights = np.zeros(len(feature_names), dtype=float)
            for feature_position, weight in explanation.as_map()[int(predicted_class)]:
                weights[int(feature_position)] = float(weight)
            repeat_weights.append(weights)
            for feature_position, weight in enumerate(weights):
                lime_rows.append(
                    {
                        "snapshot": checkpoint_path.stem,
                        "phase": phase,
                        "stream_index": int(stream_index),
                        "sample_position": int(sample_position),
                        "predicted_class": int(predicted_class),
                        "true_class": int(true_labels[sample_position]),
                        "correct": int(predicted_class == true_labels[sample_position]),
                        "selection_reason": selection_reasons[sample_position],
                        "repeat": int(repeat),
                        "feature": feature_names[feature_position],
                        "original_feature_index": int(feature_indices[feature_position]),
                        "lime_weight": float(weight),
                    }
                )
        shap_local = shap_values[sample_position, :, int(predicted_class)]
        lime_mean = np.mean(repeat_weights, axis=0)
        correlation, jaccard = rank_agreement(shap_local, lime_mean, top_k)
        stability_pairs = [
            rank_agreement(repeat_weights[first], repeat_weights[second], top_k)
            for first, second in itertools.combinations(range(lime_repeats), 2)
        ]
        finite_stability_correlations = [
            pair[0] for pair in stability_pairs if np.isfinite(pair[0])
        ]
        agreement_rows.append(
            {
                "snapshot": checkpoint_path.stem,
                "phase": phase,
                "stream_index": int(stream_index),
                "predicted_class": int(predicted_class),
                "true_class": int(true_labels[sample_position]),
                "correct": int(predicted_class == true_labels[sample_position]),
                "selection_reason": selection_reasons[sample_position],
                "shap_lime_abs_spearman": correlation,
                "shap_lime_topk_jaccard": jaccard,
                "lime_repeat_abs_spearman": float(np.mean(finite_stability_correlations)) if finite_stability_correlations else float("nan"),
                "lime_repeat_topk_jaccard": float(np.mean([pair[1] for pair in stability_pairs])) if stability_pairs else float("nan"),
            }
        )
        original_probability = float(probabilities[sample_position, predicted_class])
        for method, weights in (("shap", shap_local), ("lime", lime_mean)):
            important = sorted(top_k_set(weights, top_k))
            masked = sample.copy()
            masked[important] = background_mean[important]
            masked_probability = float(predict(masked.reshape(1, -1))[0, predicted_class])
            random_drops = []
            for _ in range(20):
                random_features = rng.choice(len(feature_names), size=len(important), replace=False)
                random_masked = sample.copy()
                random_masked[random_features] = background_mean[random_features]
                random_probability = float(predict(random_masked.reshape(1, -1))[0, predicted_class])
                random_drops.append(original_probability - random_probability)
            faithfulness_rows.append(
                {
                    "snapshot": checkpoint_path.stem,
                    "phase": phase,
                    "stream_index": int(stream_index),
                    "predicted_class": int(predicted_class),
                    "true_class": int(true_labels[sample_position]),
                    "correct": int(predicted_class == true_labels[sample_position]),
                    "selection_reason": selection_reasons[sample_position],
                    "method": method,
                    "top_k": len(important),
                    "original_probability": original_probability,
                    "masked_probability": masked_probability,
                    "important_feature_probability_drop": original_probability - masked_probability,
                    "random_feature_probability_drop_mean": float(np.mean(random_drops)),
                    "faithfulness_advantage": (original_probability - masked_probability) - float(np.mean(random_drops)),
                }
            )
    lime_seconds = time.perf_counter() - lime_start

    snapshot_dir = output_dir / checkpoint_path.stem
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        snapshot_dir / "shap_values.npz",
        shap_values=shap_values.astype(np.float32),
        explained=explained.astype(np.float32),
        stream_indices=explained_indices.astype(np.int64),
        probabilities=probabilities.astype(np.float32),
        predictions=predictions.astype(np.int64),
        true_labels=true_labels.astype(np.int64),
        selection_reasons=np.asarray(selection_reasons),
        feature_names=np.asarray(feature_names),
    )
    write_csv(global_rows, snapshot_dir / "shap_global_importance.csv")
    write_csv(lime_rows, snapshot_dir / "lime_local_weights.csv")
    write_csv(agreement_rows, snapshot_dir / "agreement_stability.csv")
    write_csv(faithfulness_rows, snapshot_dir / "faithfulness.csv")
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "snapshot": checkpoint_path.stem,
        "snapshot_labels": checkpoint["snapshot_labels"],
        "snapshot_step": int(checkpoint["snapshot_step"]),
        "phase": phase,
        "feature_names": feature_names,
        "background_stream": "same phase observations available no later than snapshot",
        "background_size": int(len(background)),
        "explained_stream_indices": explained_indices.tolist(),
        "explained_size": int(len(explained)),
        "shap_explainer": "model-agnostic KernelExplainer over complete fused probability output",
        "shap_nsamples": int(shap_nsamples),
        "shap_seconds": float(shap_seconds),
        "lime_num_samples": int(lime_num_samples),
        "lime_repeats": int(lime_repeats),
        "lime_seconds": float(lime_seconds),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--background-size", type=int, default=20)
    parser.add_argument("--explain-size", type=int, default=8)
    parser.add_argument("--shap-nsamples", type=int, default=100)
    parser.add_argument("--lime-num-samples", type=int, default=1000)
    parser.add_argument("--lime-repeats", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or (run_dir / "explainability")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = sorted((run_dir / "checkpoints").glob("*.pth"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No phase snapshots found in {run_dir / 'checkpoints'}")
    first_checkpoint = torch.load(checkpoint_paths[0], map_location="cpu", weights_only=False)
    x1, y1, x2, y2 = load_stream(first_checkpoint["run_metadata"])
    manifests = []
    for checkpoint_index, checkpoint_path in enumerate(checkpoint_paths):
        print(f"[EXPLAIN] {checkpoint_path.name}")
        manifests.append(
            analyze_snapshot(
                checkpoint_path,
                x1,
                y1,
                x2,
                y2,
                output_dir,
                args.background_size,
                args.explain_size,
                args.shap_nsamples,
                args.lime_num_samples,
                args.lime_repeats,
                args.top_k,
                args.seed + checkpoint_index * 1000,
            )
        )
    feature_metadata = first_checkpoint["run_metadata"]["feature_metadata"]
    old_features = set(int(value) for value in feature_metadata["old_only_indices"])
    shared_features = set(int(value) for value in feature_metadata["shared_indices"])
    new_features = set(int(value) for value in feature_metadata["new_only_indices"])

    global_rows = []
    agreement_rows = []
    faithfulness_rows = []
    for snapshot in manifests:
        snapshot_dir = output_dir / snapshot["snapshot"]
        for row in read_csv_rows(snapshot_dir / "shap_global_importance.csv"):
            feature_index = int(row["original_feature_index"])
            if feature_index in old_features:
                role = "obsolete_s1_only"
            elif feature_index in shared_features:
                role = "shared"
            elif feature_index in new_features:
                role = "new_s2_only"
            else:
                role = "unknown"
            row["feature_role"] = role
            global_rows.append(row)
        agreement_rows.extend(read_csv_rows(snapshot_dir / "agreement_stability.csv"))
        faithfulness_rows.extend(read_csv_rows(snapshot_dir / "faithfulness.csv"))
    write_csv(global_rows, output_dir / "shap_global_importance_all_snapshots.csv")
    write_csv(agreement_rows, output_dir / "agreement_stability_all_snapshots.csv")
    write_csv(faithfulness_rows, output_dir / "faithfulness_all_snapshots.csv")

    manifest = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "seed": int(args.seed),
        "snapshots": manifests,
        "feature_roles": {
            "obsolete_s1_only": sorted(old_features),
            "shared": sorted(shared_features),
            "new_s2_only": sorted(new_features),
        },
        "limitations": [
            "Kernel SHAP is model-agnostic and computationally expensive; a protocol-defined sample is explained.",
            "LIME depends on its perturbation distribution; repeated seeds quantify but do not eliminate instability.",
            "Correlated features can distribute attribution across substitutes.",
            "S1 and S2 have different feature sets, so cross-phase comparisons use original feature identities and explicitly distinguish shared, obsolete, and new features.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] Explainability outputs: {output_dir}")


if __name__ == "__main__":
    main()
