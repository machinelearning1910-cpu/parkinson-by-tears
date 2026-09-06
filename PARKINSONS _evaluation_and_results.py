

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# =============================================================================
# 0. PATHS / CONFIG
# =============================================================================
SEED = 2026
ROOT = Path("/content/drive/MyDrive/Parkinson_Tear_Proteomics")
RUN = ROOT / "05_PAPER_ALIGNED_137"

KG_DIR = RUN / "03_KG_preprocessing"
V2_DIR = RUN / "04_V2_training"
LORA_DIR = RUN / "05_PD_spectrum_LoRA" / "training"
EXTERNAL_ROOT = ROOT / "01_data/external_processed/PXD028811"

RESULTS = RUN / "06_evaluation"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
EXTERNAL_OUT = RESULTS / "external_PXD028811"
for p in [TABLES, FIGURES, EXTERNAL_OUT]:
    p.mkdir(parents=True, exist_ok=True)

MANIFEST_FILE = KG_DIR / "protein_KG_manifest_1011.csv"
SCALER_FILE = KG_DIR / "training_rank_scaler.joblib"
V2_MODEL_FILE = V2_DIR / "best_KG_DA_V2.pt"
V2_HISTORY_FILE = V2_DIR / "V2_training_history.csv"
V2_PRED_FILE = V2_DIR / "V2_internal_validation_predictions.csv"
LORA_MODEL_FILE = LORA_DIR / "best_PD_spectrum_LoRA_v2_137.pt"
LORA_HISTORY_FILE = LORA_DIR / "PD_spectrum_LoRA_137_training_history.csv"
LORA_PRED_FILE = LORA_DIR / "PD_spectrum_LoRA_137_internal_validation_predictions.csv"
LORA_CONFIG_FILE = LORA_DIR / "PD_spectrum_LoRA_137_training_config.json"


# =============================================================================
# 1. COMMON HELPERS
# =============================================================================
def finish(filename: str, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=400, bbox_inches="tight")
    plt.show()
    plt.close()


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    yp = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "AUROC": float(roc_auc_score(y, p)),
        "AUPRC": float(average_precision_score(y, p)),
        "Accuracy": float(accuracy_score(y, yp)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y, yp)),
        "Precision": float(precision_score(y, yp, zero_division=0)),
        "Sensitivity": float(recall_score(y, yp, zero_division=0)),
        "Specificity": float(specificity),
        "F1": float(f1_score(y, yp, zero_division=0)),
        "MCC": float(matthews_corrcoef(y, yp)),
        "Brier": float(brier_score_loss(y, p)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Threshold": float(threshold),
    }


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, p)
    return float(thresholds[np.argmax(tpr - fpr)])


def plot_confusion(cm: np.ndarray, labels: list[str], title: str, filename: str) -> None:
    plt.figure(figsize=(5.2, 4.6))
    plt.imshow(cm)
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=14)
    finish(filename, "Predicted Class", "True Class", title)


# =============================================================================
# 2. V2 INTERNAL RESULTS 
# =============================================================================
def evaluate_v2_internal() -> tuple[pd.DataFrame, pd.DataFrame, float]:
    history = pd.read_csv(V2_HISTORY_FILE)
    pred = pd.read_csv(V2_PRED_FILE)
    with open(V2_DIR / "V2_training_config.json") as f:
        config = json.load(f)
    best_epoch = int(config["best_epoch"])

    # Training / validation accuracy.
    plt.figure(figsize=(7.0, 4.7))
    plt.plot(history["epoch"], history["train_accuracy"], label="Training")
    plt.plot(history["epoch"], history["validation_accuracy"], label="Internal validation")
    plt.axvline(best_epoch, linestyle="--", linewidth=1, label=f"Selected epoch = {best_epoch}")
    plt.legend(frameon=False)
    finish("04_V2_accuracy.png", "Epoch", "Accuracy", "V2 Training and Validation Accuracy")

    # Training / validation loss.
    plt.figure(figsize=(7.0, 4.7))
    plt.plot(history["epoch"], history["train_loss"], label="Training")
    plt.plot(history["epoch"], history["validation_loss"], label="Internal validation")
    plt.axvline(best_epoch, linestyle="--", linewidth=1, label=f"Selected epoch = {best_epoch}")
    plt.legend(frameon=False)
    finish("05_V2_loss.png", "Epoch", "Loss", "V2 Training and Validation Loss")

    # Validation AUROC.
    plt.figure(figsize=(7.0, 4.7))
    plt.plot(history["epoch"], history["validation_AUROC"])
    plt.axvline(best_epoch, linestyle="--", linewidth=1)
    finish("06_V2_validation_AUROC.png", "Epoch", "AUROC", "V2 Validation AUROC")

    # Validation AUPRC.
    plt.figure(figsize=(7.0, 4.7))
    plt.plot(history["epoch"], history["validation_AUPRC"])
    plt.axvline(best_epoch, linestyle="--", linewidth=1)
    finish("07_V2_validation_AUPRC.png", "Epoch", "AUPRC", "V2 Validation AUPRC")

    y = pred["label"].astype(int).to_numpy()
    p = pred["probability_iPD"].astype(float).to_numpy()
    threshold = youden_threshold(y, p)
    metrics = binary_metrics(y, p, threshold)
    metrics["N"] = len(y)
    metrics["Selected_Epoch"] = best_epoch
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(TABLES / "Table4_V2_internal_validation.csv", index=False)

    # Performance summary bars.
    metric_names = [
        "AUROC", "AUPRC", "Accuracy", "Precision", "Sensitivity", "Specificity", "F1"
    ]
    plt.figure(figsize=(8.5, 4.8))
    plt.bar(metric_names, [metrics[m] for m in metric_names])
    plt.ylim(0, 1.05)
    plt.xticks(rotation=25, ha="right")
    finish("08_V2_internal_performance.png", "Metric", "Score", "V2 Internal Validation Performance")

    cm = np.array([[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]])
    plot_confusion(cm, ["Control", "iPD"], "V2 Internal Validation Confusion Matrix", "09_V2_confusion_matrix.png")

    # Center-specific performance uses threshold 0.5 exactly as in the supplied code.
    center_rows = []
    for center in ["CRUCES", "DONOSTIA"]:
        sub = pred[pred["center"] == center]
        yy = sub["label"].astype(int).to_numpy()
        pp = sub["probability_iPD"].astype(float).to_numpy()
        mm = binary_metrics(yy, pp, threshold=0.5)
        mm["Center"] = center
        mm["N"] = len(sub)
        center_rows.append(mm)
    center = pd.DataFrame(center_rows)
    center.to_csv(TABLES / "Table5_V2_center_specific.csv", index=False)

    for metric, filename in [
        ("AUROC", "10_center_AUROC.png"),
        ("Accuracy", "11_center_accuracy.png"),
        ("F1", "12_center_F1.png"),
    ]:
        plt.figure(figsize=(5.5, 4.3))
        plt.bar(center["Center"], center[metric])
        plt.ylim(0, 1.05)
        finish(filename, "Center", metric, f"Center-Specific {metric}")

    return metrics_df, center, threshold


# =============================================================================
# 3. LoRA PARAMETER EFFICIENCY + TRAINING TRAJECTORIES
# =============================================================================
def evaluate_lora_internal() -> tuple[pd.DataFrame, dict]:
    history = pd.read_csv(LORA_HISTORY_FILE)
    pred = pd.read_csv(LORA_PRED_FILE)
    with open(LORA_CONFIG_FILE) as f:
        config = json.load(f)

    assert config["train_n"] == 137
    assert config["train_LRRK2"] == 7
    assert config["train_E46K"] == 0
    assert config["internal_E46K_used"] is False

    params = pd.DataFrame(
        [
            {
                "Parameter category": "Frozen V2 parameters",
                "Number of parameters": config["frozen_parameters"],
                "Percentage": 100 - config["trainable_fraction_percent"],
            },
            {
                "Parameter category": "Trainable LoRA parameters",
                "Number of parameters": config["LoRA_trainable_parameters"],
                "Percentage": config["trainable_fraction_percent"],
            },
            {
                "Parameter category": "Total parameters",
                "Number of parameters": config["total_parameters"],
                "Percentage": 100.0,
            },
        ]
    )
    params.to_csv(TABLES / "Table6_LoRA_parameter_efficiency.csv", index=False)

    plt.figure(figsize=(5.8, 4.4))
    plt.bar(
        ["Frozen V2", "Trainable LoRA"],
        [config["frozen_parameters"], config["LoRA_trainable_parameters"]],
    )
    finish("13_parameter_efficiency.png", "Parameter Group", "Number of Parameters", "PD-Spectrum LoRA Parameter Efficiency")

    # Training/validation AUROC trajectory.
    plt.figure(figsize=(7.0, 4.7))
    valid = history["epoch"] > 0
    plt.plot(history.loc[valid, "epoch"], history.loc[valid, "train_AUROC"], label="Training")
    plt.plot(history["epoch"], history["validation_AUROC"], label="Internal validation")
    plt.axvline(config["best_epoch"], linestyle="--", linewidth=1, label=f"Selected epoch = {config['best_epoch']}")
    plt.legend(frameon=False)
    finish("15_LoRA_AUROC_trajectory.png", "Epoch", "AUROC", "PD-Spectrum LoRA AUROC")

    # Training/validation AUPRC trajectory.
    plt.figure(figsize=(7.0, 4.7))
    plt.plot(history.loc[valid, "epoch"], history.loc[valid, "train_AUPRC"], label="Training")
    plt.plot(history["epoch"], history["validation_AUPRC"], label="Internal validation")
    plt.axvline(config["best_epoch"], linestyle="--", linewidth=1, label=f"Selected epoch = {config['best_epoch']}")
    plt.legend(frameon=False)
    finish("16_LoRA_AUPRC_trajectory.png", "Epoch", "AUPRC", "PD-Spectrum LoRA AUPRC")

    # Internal LoRA metrics at its locked threshold (reported separately from V2).
    y = pred["label"].astype(int).to_numpy()
    p = pred["probability_PD_spectrum"].astype(float).to_numpy()
    threshold = float(config["locked_internal_threshold"])
    metrics = binary_metrics(y, p, threshold)
    metrics["N"] = len(y)
    metrics["Best_Epoch"] = config["best_epoch"]
    pd.DataFrame([metrics]).to_csv(TABLES / "LoRA_internal_validation_metrics.csv", index=False)

    return params, config


# =============================================================================
# 4. V2 LEARNED WEIGHT DISTRIBUTION
# =============================================================================
class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


class KGGatedDomainModel(nn.Module):
    def __init__(self, n_features: int, prior: torch.Tensor):
        super().__init__()
        self.register_buffer("kg_prior", prior.clone())
        self.gate_residual = nn.Parameter(torch.zeros(n_features))
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.25),
        )
        self.disease_head = nn.Linear(64, 1)
        self.domain_head = nn.Sequential(
            nn.Linear(64, 32), nn.GELU(), nn.Dropout(0.15), nn.Linear(32, 2)
        )

    def protein_gate(self):
        return 1.0 + 0.25 * self.kg_prior + 0.10 * torch.tanh(self.gate_residual)

    def forward(self, x, alpha=0.0):
        x = x * self.protein_gate()
        z = self.encoder(x)
        return self.disease_head(z).squeeze(1), self.domain_head(GRL.apply(z, alpha)), z


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.A = nn.Parameter(torch.empty(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))

    def forward(self, x):
        delta = F.linear(F.linear(self.dropout(x), self.A), self.B)
        return self.base(x) + self.scale * delta


def make_lora_model(manifest: pd.DataFrame) -> KGGatedDomainModel:
    prior = torch.tensor(manifest["PD_KG_prior"].to_numpy(np.float32), dtype=torch.float32)
    model = KGGatedDomainModel(1011, prior)
    model.encoder[0] = LoRALinear(model.encoder[0], 8, 16, 0.10)
    model.encoder[4] = LoRALinear(model.encoder[4], 4, 8, 0.10)
    model.disease_head = LoRALinear(model.disease_head, 2, 4, 0.05)
    return model


def plot_v2_weight_distribution() -> None:
    manifest = pd.read_csv(MANIFEST_FILE)
    prior = torch.tensor(manifest["PD_KG_prior"].to_numpy(np.float32), dtype=torch.float32)
    model = KGGatedDomainModel(1011, prior)
    model.load_state_dict(torch.load(V2_MODEL_FILE, map_location="cpu"), strict=True)
    values = []
    for name, p in model.named_parameters():
        if p.ndim >= 2:
            values.append(p.detach().cpu().numpy().ravel())
    weights = np.concatenate(values)
    plt.figure(figsize=(6.5, 4.7))
    plt.hist(weights, bins=60)
    finish("17_V2_weight_distribution.png", "Learned Weight Value", "Frequency", "Distribution of Learned V2 Weights")


# =============================================================================
# 5. STRICT EXTERNAL VALIDATION ON ALL 54 PXD028811 PARTICIPANTS
# =============================================================================
def locate_external_matrix() -> Path:
    preferred = (
        EXTERNAL_ROOT / "model_ready" / "PXD028811_raw_area_internal_feature_space.parquet"
    )
    if preferred.exists():
        return preferred
    candidates = [
        p for p in EXTERNAL_ROOT.rglob("*.parquet")
        if any(k in p.name.lower() for k in ("raw_area", "internal_feature", "model_ready"))
    ] if EXTERNAL_ROOT.exists() else []
    if not candidates:
        raise FileNotFoundError("Could not locate the processed PXD028811 external matrix.")
    return candidates[0]


def bootstrap_external(y: np.ndarray, p: np.ndarray, threshold: float, n_boot: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    control_idx = np.where(y == 0)[0]
    pd_idx = np.where(y == 1)[0]
    rows = []
    for _ in range(n_boot):
        idx = np.concatenate(
            [
                rng.choice(control_idx, len(control_idx), replace=True),
                rng.choice(pd_idx, len(pd_idx), replace=True),
            ]
        )
        yy, pp = y[idx], p[idx]
        rows.append(binary_metrics(yy, pp, threshold))
    return pd.DataFrame(rows)


def evaluate_external() -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(MANIFEST_FILE)
    proteins = manifest["protein"].astype(str).tolist()
    scaler = joblib.load(SCALER_FILE)
    with open(LORA_CONFIG_FILE) as f:
        config = json.load(f)

    # Critical development-cohort assertions.
    assert config["train_n"] == 137
    assert config["train_LRRK2"] == 7
    assert config["train_E46K"] == 0
    locked_threshold = float(config["locked_internal_threshold"])

    matrix = locate_external_matrix()
    external = pd.read_parquet(matrix).copy()
    external["sample_id"] = external["sample_id"].astype(str).str.strip()
    external = external.drop_duplicates("sample_id").reset_index(drop=True)
    assert len(external) == 54

    # Supplied final code mapping: Ctrl* = control; A*/Lag* = PD-side.
    external["external_group"] = np.where(
        external["sample_id"].str.upper().str.startswith("CTRL"),
        "Control",
        "PD-spectrum",
    )
    external["label"] = external["external_group"].map({"Control": 0, "PD-spectrum": 1}).astype(int)
    assert (external["label"] == 0).sum() == 27
    assert (external["label"] == 1).sum() == 27

    # The external PD-spectrum is 24 iPD + 3 E46K-SNCA. Those three external
    # E46K cases are retained because the paper explicitly evaluates them.
    # No external subtype is used to fit or adapt the model.

    available = [p for p in proteins if p in external.columns]
    for protein in proteins:
        if protein not in external.columns:
            external[protein] = np.nan
    X_raw = external[proteins].apply(pd.to_numeric, errors="coerce")
    observed = X_raw.notna().any(axis=0)

    # Within-participant percentile ranks only; no external cohort fitting.
    X_rank = X_raw.rank(axis=1, method="average", pct=True, na_option="keep").to_numpy(np.float64)
    missing = ~np.isfinite(X_rank)
    reference = np.broadcast_to(scaler.mean_, X_rank.shape)
    X_rank[missing] = reference[missing]
    X_ext = scaler.transform(pd.DataFrame(X_rank, columns=proteins)).astype(np.float32)
    assert np.isfinite(X_ext).all()

    feature_audit = pd.DataFrame(
        {
            "protein": proteins,
            "available_in_external": observed.values,
            "PD_KG_prior": manifest["PD_KG_prior"].values,
        }
    )
    feature_audit.to_csv(EXTERNAL_OUT / "PXD028811_feature_transportability.csv", index=False)

    model = make_lora_model(manifest)
    model.load_state_dict(torch.load(LORA_MODEL_FILE, map_location="cpu"), strict=True)
    for param in model.parameters():
        param.requires_grad = False
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    with torch.no_grad():
        logits, _, embeddings = model(torch.tensor(X_ext, dtype=torch.float32, device=device), alpha=0.0)
        probabilities = torch.sigmoid(logits).cpu().numpy()

    y = external["label"].to_numpy(int)
    metrics = binary_metrics(y, probabilities, locked_threshold)
    metrics.update(
        {
            "Dataset": "PXD028811",
            "Endpoint": "Control vs PD-spectrum",
            "N": 54,
            "Control_n": 27,
            "PD_spectrum_n": 27,
            "PD_spectrum_external_definition": "24 iPD + 3 E46K-SNCA",
            "Development_train_n": 137,
            "Development_definition": "60 iPD + 7 LRRK2-PD vs 70 controls",
            "Internal_E46K_used_for_development": False,
            "Externally_observed_proteins": int(observed.sum()),
            "Neutral_imputed_proteins": int(1011 - observed.sum()),
            "External_training": False,
            "External_LoRA_tuning": False,
            "External_domain_adaptation": False,
            "External_threshold_optimization": False,
            "External_feature_selection": False,
        }
    )
    result_df = pd.DataFrame([metrics])
    result_df.to_csv(EXTERNAL_OUT / "Table7_PXD028811_external_results.csv", index=False)

    external["probability_PD_spectrum"] = probabilities
    external["prediction"] = (probabilities >= locked_threshold).astype(int)
    external.to_csv(EXTERNAL_OUT / "PXD028811_external_predictions.csv", index=False)

    emb = pd.DataFrame(
        embeddings.cpu().numpy(), columns=[f"embedding_{i+1}" for i in range(64)]
    )
    emb.insert(0, "sample_id", external["sample_id"].values)
    emb.to_parquet(EXTERNAL_OUT / "PXD028811_frozen_embeddings.parquet", index=False)

    protocol = {
        "development_train_n": 137,
        "development_PD_spectrum": "iPD + LRRK2-PD",
        "internal_E46K_used": False,
        "external_n": 54,
        "external_PD_spectrum": "24 iPD + 3 E46K-SNCA",
        "locked_internal_threshold": locked_threshold,
        "external_training": False,
        "external_LoRA_tuning": False,
        "external_domain_adaptation": False,
        "external_feature_selection": False,
        "external_threshold_optimization": False,
    }
    with open(EXTERNAL_OUT / "PXD028811_locked_protocol.json", "w") as f:
        json.dump(protocol, f, indent=2)

    # Bootstrap 95% confidence intervals with the threshold kept fixed.
    boot = bootstrap_external(y, probabilities, locked_threshold, 2000)
    boot.to_csv(EXTERNAL_OUT / "PXD028811_bootstrap_2000.csv", index=False)
    ci_rows = []
    for metric in [
        "AUROC", "AUPRC", "Accuracy", "Balanced_Accuracy", "Precision",
        "Sensitivity", "Specificity", "F1", "MCC", "Brier",
    ]:
        low, high = np.percentile(boot[metric], [2.5, 97.5])
        ci_rows.append(
            {
                "Metric": metric,
                "Estimate": metrics[metric],
                "CI95_low": low,
                "CI95_high": high,
            }
        )
    ci = pd.DataFrame(ci_rows)
    ci.to_csv(EXTERNAL_OUT / "PXD028811_bootstrap_95CI.csv", index=False)

    # External result figure.
    names = ["AUROC", "AUPRC", "Accuracy", "Precision", "Sensitivity", "Specificity", "F1"]
    plt.figure(figsize=(8.5, 4.8))
    plt.bar(names, [metrics[m] for m in names])
    plt.ylim(0, 1.05)
    plt.xticks(rotation=25, ha="right")
    finish("18_external_performance.png", "Metric", "Score", "PXD028811 Independent External Validation")

    cm = np.array([[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]])
    plot_confusion(cm, ["Control", "PD-spectrum"], "PXD028811 External Confusion Matrix", "19_external_confusion_matrix.png")

    # ROC and PR curves are evaluation only; they do not change the threshold.
    fpr, tpr, _ = roc_curve(y, probabilities)
    plt.figure(figsize=(5.5, 5.0))
    plt.plot(fpr, tpr, label=f"AUROC = {metrics['AUROC']:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.legend(frameon=False)
    finish("20_external_ROC.png", "False Positive Rate", "True Positive Rate", "External ROC Curve")

    precision, recall, _ = precision_recall_curve(y, probabilities)
    plt.figure(figsize=(5.5, 5.0))
    plt.plot(recall, precision, label=f"AUPRC = {metrics['AUPRC']:.3f}")
    plt.legend(frameon=False)
    finish("21_external_PR.png", "Recall", "Precision", "External Precision-Recall Curve")

    return result_df, ci


# =============================================================================
# 6.  REPORT 
# =============================================================================
def paper_alignment_report(
    v2_metrics: pd.DataFrame,
    center_metrics: pd.DataFrame,
    lora_config: dict,
    external_results: pd.DataFrame,
) -> pd.DataFrame:
    """Compare computed outputs to paper values; never overwrite computed results."""
    rows = []

    def add(section: str, item: str, computed: float, paper: float, tol: float = 0.005):
        rows.append(
            {
                "Section": section,
                "Item": item,
                "Computed": float(computed),
                "Paper": float(paper),
                "Absolute_Difference": abs(float(computed) - float(paper)),
                "Tolerance": tol,
                "Aligned": abs(float(computed) - float(paper)) <= tol,
            }
        )

    v = v2_metrics.iloc[0]
    for metric, expected in {
        "AUROC": 0.908,
        "AUPRC": 0.877,
        "Accuracy": 0.909,
        "Precision": 0.933,
        "Sensitivity": 0.875,
        "Specificity": 0.941,
        "Balanced_Accuracy": 0.908,
        "F1": 0.903,
    }.items():
        add("V2 internal", metric, v[metric], expected)
    for item, expected in {"TN": 16, "FP": 1, "FN": 2, "TP": 14}.items():
        add("V2 internal confusion", item, v[item], expected, tol=0.0)

    for center, expected in {
        "CRUCES": {"AUROC": 0.930, "Accuracy": 0.950, "F1": 0.947},
        "DONOSTIA": {"AUROC": 0.881, "Accuracy": 0.846, "F1": 0.833},
    }.items():
        c = center_metrics[center_metrics["Center"] == center].iloc[0]
        for metric, value in expected.items():
            add(f"V2 center {center}", metric, c[metric], value)

    add("LoRA architecture", "Total parameters", lora_config["total_parameters"], 151408, tol=0.0)
    add("LoRA architecture", "Trainable parameters", lora_config["LoRA_trainable_parameters"], 10010, tol=0.0)
    add("LoRA cohort", "Training N", lora_config["train_n"], 137, tol=0.0)
    add("LoRA cohort", "LRRK2 N", lora_config["train_LRRK2"], 7, tol=0.0)
    add("LoRA cohort", "Internal E46K N", lora_config["train_E46K"], 0, tol=0.0)

    e = external_results.iloc[0]
    add("External", "Locked threshold", e["Threshold"], 0.0832, tol=0.005)
    for metric, expected in {
        "AUROC": 0.890,
        "AUPRC": 0.870,
        "Accuracy": 0.889,
        "Balanced_Accuracy": 0.889,
        "Precision": 0.862,
        "Sensitivity": 0.926,
        "Specificity": 0.852,
        "F1": 0.893,
    }.items():
        add("External", metric, e[metric], expected)
    for item, expected in {"TN": 23, "FP": 4, "FN": 2, "TP": 25}.items():
        add("External confusion", item, e[item], expected, tol=0.0)

    report = pd.DataFrame(rows)
    report.to_csv(TABLES / "paper_alignment_report.csv", index=False)
    return report


# =============================================================================
# 7. MAIN
# =============================================================================
def main() -> None:
    required = [
        MANIFEST_FILE, SCALER_FILE, V2_MODEL_FILE, V2_HISTORY_FILE, V2_PRED_FILE,
        LORA_MODEL_FILE, LORA_HISTORY_FILE, LORA_PRED_FILE, LORA_CONFIG_FILE,
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing methodology output:\n{p}\nRun 02_methodology_training.py first."
            )

    v2_metrics, center_metrics, _ = evaluate_v2_internal()
    _, lora_config = evaluate_lora_internal()
    plot_v2_weight_distribution()
    external_results, _ = evaluate_external()
    alignment = paper_alignment_report(v2_metrics, center_metrics, lora_config, external_results)

    print("\n" + "=" * 90)
    print("PAPER-ALIGNED EVALUATION COMPLETE")
    print("=" * 90)
    print(f"137-person LoRA cohort verified: {lora_config['train_n'] == 137}")
    print(f"Internal E46K used in training : {lora_config['train_E46K']}")
    print(f"Alignment checks passed       : {int(alignment['Aligned'].sum())}/{len(alignment)}")
    if not alignment["Aligned"].all():
        print("\nIMPORTANT: Some recomputed values differ from the manuscript.")
        print("Do not copy old 138-person results into the corrected 137-person pipeline.")
        print("Inspect paper_alignment_report.csv and update the manuscript only from the rerun outputs.")
    print(f"\nTables : {TABLES}")
    print(f"Figures: {FIGURES}")
    print(f"External outputs: {EXTERNAL_OUT}")


if __name__ == "__main__":
    main()
