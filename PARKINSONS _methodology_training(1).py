

from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# 0. CONFIGURATION
# =============================================================================
SEED = 2026
ROOT = Path("/content/drive/MyDrive/Parkinson_Tear_Proteomics")
RUN = ROOT / "05_PAPER_ALIGNED_137"
DATASET = RUN / "01_dataset"
MANIFESTS = DATASET / "manifests"

SPLIT_OUT = RUN / "02_internal_split"
KG_OUT = RUN / "03_KG_preprocessing"
V2_OUT = RUN / "04_V2_training"
LORA_DATA_OUT = RUN / "05_PD_spectrum_LoRA" / "data"
LORA_TRAIN_OUT = RUN / "05_PD_spectrum_LoRA" / "training"
for p in [SPLIT_OUT, KG_OUT, V2_OUT, LORA_DATA_OUT, LORA_TRAIN_OUT]:
    p.mkdir(parents=True, exist_ok=True)

INTERNAL_163 = ROOT / "01_data/splits/PXD068184/binary_163_before_split.parquet"
LRRK2_7 = MANIFESTS / "PXD068184_LRRK2_7_with_proteins.parquet"
E46K_EXCLUDED = MANIFESTS / "PXD068184_E46K_1_EXCLUDED_with_proteins.parquet"

KG_EMBED_DIR = ROOT / "01_data/kg/PrimeKG/core_model_KG/pretrained_complex"
KG_MANIFEST_SOURCE = KG_EMBED_DIR / "protein_embedding_manifest.csv"
KG_EMBED_SOURCE = KG_EMBED_DIR / "protein_feature_embeddings_128d.npy"
KG_MASK_SOURCE = KG_EMBED_DIR / "protein_feature_mapped_mask.npy"
PD_PRIOR_SOURCE = (
    ROOT
    / "01_data/kg/PrimeKG/PD_aware_KG/trained_PD_node"
    / "measured_feature_PD_relevance_130ep.csv"
)


# =============================================================================
# 1. REPRODUCIBILITY
# =============================================================================
def set_reproducibility(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Reproducible CuDNN behavior. Deterministic algorithms are not forced globally
    # because some Colab/CUDA kernels may otherwise raise at runtime.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def mount_drive_if_colab() -> None:
    try:
        from google.colab import drive  # type: ignore
    except Exception:
        return
    drive.mount("/content/drive", force_remount=False)


# =============================================================================
# 2.  130 / 33 PARTICIPANT SPLIT
# =============================================================================
def make_fixed_split() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if not INTERNAL_163.exists():
        raise FileNotFoundError(f"Missing canonical 163-person matrix:\n{INTERNAL_163}")

    df = pd.read_parquet(INTERNAL_163).copy()
    required = ["sample_id", "center", "group", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Internal matrix is missing columns: {missing}")

    assert len(df) == 163
    assert df["sample_id"].astype(str).is_unique
    assert set(df["group"].astype(str)) == {"control", "iPD"}
    assert set(df["center"].astype(str)) == {"CRUCES", "DONOSTIA"}

    df["sample_id"] = df["sample_id"].astype(str).str.strip()
    df["stratum"] = df["center"].astype(str) + "_" + df["label"].astype(int).astype(str)

    train, val = train_test_split(
        df,
        test_size=33,
        stratify=df["stratum"],
        random_state=SEED,
    )
    train = train.drop(columns="stratum").copy().reset_index(drop=True)
    val = val.drop(columns="stratum").copy().reset_index(drop=True)
    train["split"] = "train"
    val["split"] = "internal_validation"

    assert len(train) == 130
    assert len(val) == 33
    assert set(train["sample_id"]).isdisjoint(set(val["sample_id"]))

    # Exact composition reported in the paper.
    expected_train = {
        ("CRUCES", "control"): 42,
        ("CRUCES", "iPD"): 38,
        ("DONOSTIA", "control"): 28,
        ("DONOSTIA", "iPD"): 22,
    }
    expected_val = {
        ("CRUCES", "control"): 10,
        ("CRUCES", "iPD"): 10,
        ("DONOSTIA", "control"): 7,
        ("DONOSTIA", "iPD"): 6,
    }
    for (center, group), n in expected_train.items():
        assert ((train["center"] == center) & (train["group"] == group)).sum() == n
    for (center, group), n in expected_val.items():
        assert ((val["center"] == center) & (val["group"] == group)).sum() == n

    train["domain_label"] = train["center"].map({"CRUCES": 0, "DONOSTIA": 1}).astype(int)
    val["domain_label"] = val["center"].map({"CRUCES": 0, "DONOSTIA": 1}).astype(int)

    metadata = {"sample_id", "center", "group", "label", "split", "domain_label"}
    protein_cols = [
        c for c in train.columns
        if c not in metadata and pd.api.types.is_numeric_dtype(train[c])
    ]
    assert len(protein_cols) == 1011, f"Expected 1,011 proteins, found {len(protein_cols)}"
    assert protein_cols == [
        c for c in val.columns
        if c not in metadata and pd.api.types.is_numeric_dtype(val[c])
    ]

    pd.concat([train, val], ignore_index=True)[
        ["sample_id", "center", "group", "label", "split", "domain_label"]
    ].to_csv(SPLIT_OUT / "PXD068184_train_130_validation_33.csv", index=False)

    return train, val, protein_cols


# =============================================================================
# 3. LEAKAGE-CONTROLLED PREPROCESSING + PRIMEKG PRIOR
# =============================================================================
def percentile_rank(frame: pd.DataFrame, proteins: list[str]) -> pd.DataFrame:
    return frame[proteins].rank(axis=1, method="average", pct=True, na_option="keep")


def prepare_kg_inputs(
    train: pd.DataFrame,
    val: pd.DataFrame,
    proteins: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Train-fitted preprocessing only.
    train_rank = percentile_rank(train, proteins)
    val_rank = percentile_rank(val, proteins)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(train_rank)
    Xva = scaler.transform(val_rank)
    joblib.dump(scaler, KG_OUT / "training_rank_scaler.joblib")

    # Verify the existing PrimeKG feature ordering.
    kg_manifest = pd.read_csv(KG_MANIFEST_SOURCE)
    kg_embeddings = np.load(KG_EMBED_SOURCE)
    kg_mapped = np.load(KG_MASK_SOURCE).astype(bool)
    assert "feature_name" in kg_manifest.columns
    assert kg_manifest["feature_name"].astype(str).tolist() == proteins
    assert kg_embeddings.shape[0] == 1011
    assert len(kg_mapped) == 1011

    pd_info = pd.read_csv(PD_PRIOR_SOURCE).drop_duplicates("feature_name")
    prior = pd.DataFrame({"feature_name": proteins}).merge(
        pd_info, on="feature_name", how="left"
    )
    pd_rel = pd.to_numeric(prior.get("PD_relevance", 0), errors="coerce").fillna(0).to_numpy()
    direct_col = next(
        (
            c for c in [
                "direct_PD",
                "direct_PD_association",
                "direct_parkinson_association",
            ]
            if c in prior.columns
        ),
        None,
    )
    direct = (
        prior[direct_col].fillna(False).astype(bool).to_numpy()
        if direct_col
        else np.zeros(1011, dtype=bool)
    )

    kg_prior = 0.85 * pd_rel + 0.15 * direct.astype(float)
    if kg_prior.max() > kg_prior.min():
        kg_prior = (kg_prior - kg_prior.min()) / (kg_prior.max() - kg_prior.min())

    model_manifest = pd.DataFrame(
        {
            "protein": proteins,
            "PD_KG_prior": kg_prior.astype(np.float32),
            "PrimeKG_mapped": kg_mapped,
        }
    )
    model_manifest.to_csv(KG_OUT / "protein_KG_manifest_1011.csv", index=False)

    def pack(source: pd.DataFrame, X: np.ndarray) -> pd.DataFrame:
        meta = source[
            ["sample_id", "center", "group", "label", "domain_label"]
        ].reset_index(drop=True)
        return pd.concat([meta, pd.DataFrame(X, columns=proteins)], axis=1)

    train_ready = pack(train, Xtr.astype(np.float32))
    val_ready = pack(val, Xva.astype(np.float32))
    train_ready.to_parquet(KG_OUT / "train_130_model_ready.parquet", index=False)
    val_ready.to_parquet(KG_OUT / "validation_33_model_ready.parquet", index=False)

    audit = {
        "train_n": 130,
        "validation_n": 33,
        "protein_features": 1011,
        "rank_transform": "within-participant percentile rank",
        "scaler_fit_on": "original 130 development-training participants only",
        "KG_prior": "0.85*PD_relevance + 0.15*direct_PD, min-max normalized",
    }
    with open(KG_OUT / "preprocessing_KG_audit.json", "w") as f:
        json.dump(audit, f, indent=2)

    return train_ready, val_ready, model_manifest


# =============================================================================
# 4. MODEL DEFINITIONS
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
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(32, 2),
        )

    def protein_gate(self) -> torch.Tensor:
        return 1.0 + 0.25 * self.kg_prior + 0.10 * torch.tanh(self.gate_residual)

    def forward(self, x: torch.Tensor, alpha: float = 0.0):
        x = x * self.protein_gate()
        z = self.encoder(x)
        disease = self.disease_head(z).squeeze(1)
        domain = self.domain_head(GRL.apply(z, alpha))
        return disease, domain, z


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.A = nn.Parameter(torch.empty(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=np.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = F.linear(F.linear(self.dropout(x), self.A), self.B)
        return self.base(x) + self.scale * delta


def add_paper_lora(model: KGGatedDomainModel) -> KGGatedDomainModel:
    model.encoder[0] = LoRALinear(model.encoder[0], rank=8, alpha=16, dropout=0.10)
    model.encoder[4] = LoRALinear(model.encoder[4], rank=4, alpha=8, dropout=0.10)
    model.disease_head = LoRALinear(model.disease_head, rank=2, alpha=4, dropout=0.05)
    return model


# =============================================================================
# 5. METRIC HELPERS
# =============================================================================
def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "AUROC": float(roc_auc_score(y, p)),
        "AUPRC": float(average_precision_score(y, p)),
        "Accuracy": float(accuracy_score(y, pred)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y, pred)),
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "Recall": float(recall_score(y, pred, zero_division=0)),
        "Specificity": float(specificity),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y, pred)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


# =============================================================================
# 6. V2 KNOWLEDGE-GUIDED + DOMAIN-ADAPTIVE TRAINING
# =============================================================================
def train_v2(
    train: pd.DataFrame,
    val: pd.DataFrame,
    manifest: pd.DataFrame,
    device: torch.device,
) -> None:
    proteins = manifest["protein"].tolist()
    Xtr = torch.tensor(train[proteins].to_numpy(np.float32), dtype=torch.float32)
    ytr = torch.tensor(train["label"].to_numpy(np.float32), dtype=torch.float32)
    dtr = torch.tensor(train["domain_label"].to_numpy(np.int64), dtype=torch.long)
    Xva = torch.tensor(val[proteins].to_numpy(np.float32), dtype=torch.float32, device=device)
    yva = torch.tensor(val["label"].to_numpy(np.float32), dtype=torch.float32, device=device)
    dva = torch.tensor(val["domain_label"].to_numpy(np.int64), dtype=torch.long, device=device)

    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        TensorDataset(Xtr, ytr, dtr),
        batch_size=16,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    prior = torch.tensor(manifest["PD_KG_prior"].to_numpy(np.float32), dtype=torch.float32)
    model = KGGatedDomainModel(1011, prior).to(device)

    pos_weight = torch.tensor(
        (train["label"] == 0).sum() / (train["label"] == 1).sum(),
        dtype=torch.float32,
        device=device,
    )
    task_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    domain_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)

    EPOCHS, WARMUP, MIN_LR = 150, 20, 5e-6
    LAMBDA_DA_MAX, LAMBDA_KG = 0.08, 0.02
    MIN_EPOCHS, PATIENCE = 50, 30

    def lr_for_epoch(epoch: int) -> float:
        if epoch <= WARMUP:
            return 1e-5 + (1e-4 - 1e-5) * epoch / WARMUP
        progress = (epoch - WARMUP) / (EPOCHS - WARMUP)
        return MIN_LR + 0.5 * (1e-4 - MIN_LR) * (1 + np.cos(np.pi * progress))

    history: list[dict] = []
    best_auc = -np.inf
    best_auprc = -np.inf
    best_epoch = 0
    wait = 0

    for epoch in range(1, EPOCHS + 1):
        lr = lr_for_epoch(epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if epoch <= 15:
            alpha, lambda_da = 0.0, 0.0
        else:
            progress = min(1.0, (epoch - 15) / 45)
            alpha = float(2 / (1 + np.exp(-6 * progress)) - 1)
            lambda_da = LAMBDA_DA_MAX * progress

        model.train()
        total_losses, task_losses, domain_losses, kg_losses = [], [], [], []
        train_probs, train_true = [], []

        for xb, yb, db in loader:
            xb, yb, db = xb.to(device), yb.to(device), db.to(device)
            optimizer.zero_grad()
            logits, domain_logits, _ = model(xb, alpha)
            l_task = task_loss_fn(logits, yb)
            l_domain = domain_loss_fn(domain_logits, db)
            l_kg = model.gate_residual.pow(2).mean()
            loss = l_task + lambda_da * l_domain + LAMBDA_KG * l_kg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_losses.append(loss.item())
            task_losses.append(l_task.item())
            domain_losses.append(l_domain.item())
            kg_losses.append(l_kg.item())
            train_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())
            train_true.extend(yb.detach().cpu().numpy())

        train_probs = np.asarray(train_probs)
        train_true = np.asarray(train_true).astype(int)
        train_acc = accuracy_score(train_true, (train_probs >= 0.5).astype(int))
        train_f1 = f1_score(train_true, (train_probs >= 0.5).astype(int), zero_division=0)

        model.eval()
        with torch.no_grad():
            val_logits, val_domain_logits, _ = model(Xva, alpha=0.0)
            val_loss = task_loss_fn(val_logits, yva).item()
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_domain_pred = val_domain_logits.argmax(1).cpu().numpy()

        y_val = val["label"].to_numpy(int)
        val_metrics = binary_metrics(y_val, val_probs, threshold=0.5)
        val_auc = val_metrics["AUROC"]
        val_auprc = val_metrics["AUPRC"]
        domain_acc = accuracy_score(val["domain_label"], val_domain_pred)

        history.append(
            {
                "epoch": epoch,
                "lr": lr,
                "train_loss": float(np.mean(total_losses)),
                "train_task_loss": float(np.mean(task_losses)),
                "train_domain_loss": float(np.mean(domain_losses)),
                "train_KG_loss": float(np.mean(kg_losses)),
                "validation_loss": float(val_loss),
                "train_accuracy": float(train_acc),
                "validation_accuracy": val_metrics["Accuracy"],
                "train_F1": float(train_f1),
                "validation_F1": val_metrics["F1"],
                "validation_precision": val_metrics["Precision"],
                "validation_recall": val_metrics["Recall"],
                "validation_AUROC": val_auc,
                "validation_AUPRC": val_auprc,
                "domain_accuracy": float(domain_acc),
                "GRL_alpha": alpha,
                "lambda_DA": lambda_da,
            }
        )

        # Primary checkpoint criterion = validation AUROC; AUPRC tie-breaker.
        improved = (
            val_auc > best_auc + 1e-4
            or (abs(val_auc - best_auc) <= 1e-4 and val_auprc > best_auprc + 1e-4)
        )
        if improved:
            best_auc, best_auprc, best_epoch, wait = val_auc, val_auprc, epoch, 0
            torch.save(model.state_dict(), V2_OUT / "best_KG_DA_V2.pt")
        else:
            wait += 1

        if epoch == 1 or epoch % 10 == 0 or improved:
            print(
                f"V2 {epoch:03d} | train loss {np.mean(total_losses):.4f} | "
                f"val loss {val_loss:.4f} | AUROC {val_auc:.4f} | AUPRC {val_auprc:.4f}"
            )
        if epoch >= MIN_EPOCHS and wait >= PATIENCE:
            print(f"V2 early stopping at epoch {epoch}")
            break

    pd.DataFrame(history).to_csv(V2_OUT / "V2_training_history.csv", index=False)

    model.load_state_dict(torch.load(V2_OUT / "best_KG_DA_V2.pt", map_location=device))
    model.eval()
    with torch.no_grad():
        logits, domain_logits, embeddings = model(Xva, alpha=0.0)
        probs = torch.sigmoid(logits).cpu().numpy()
        domain_pred = domain_logits.argmax(1).cpu().numpy()

    pred = val[["sample_id", "center", "group", "label", "domain_label"]].copy()
    pred["probability_iPD"] = probs
    pred["predicted_domain"] = domain_pred
    pred.to_csv(V2_OUT / "V2_internal_validation_predictions.csv", index=False)

    emb = pd.DataFrame(
        embeddings.cpu().numpy(), columns=[f"embedding_{i+1}" for i in range(64)]
    )
    emb.insert(0, "sample_id", val["sample_id"].values)
    emb.to_parquet(V2_OUT / "V2_validation_embeddings.parquet", index=False)

    config = {
        "seed": SEED,
        "input_features": 1011,
        "encoder": [1011, 128, 64],
        "domain_head": [64, 32, 2],
        "train_n": 130,
        "validation_n": 33,
        "optimizer": "AdamW",
        "batch_size": 16,
        "max_epochs": 150,
        "warmup_epochs": 20,
        "peak_learning_rate": 1e-4,
        "weight_decay": 5e-4,
        "max_domain_loss_coefficient": 0.08,
        "KG_residual_coefficient": 0.02,
        "best_epoch": int(best_epoch),
        "best_validation_AUROC": float(best_auc),
        "best_validation_AUPRC": float(best_auprc),
        "external_dataset_used": False,
    }
    with open(V2_OUT / "V2_training_config.json", "w") as f:
        json.dump(config, f, indent=2)


# =============================================================================
# 7. EXACTLY SEVEN LRRK2 CASES 
# =============================================================================
def build_lora_137_cohort() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for p in [LRRK2_7, E46K_EXCLUDED, KG_OUT / "train_130_model_ready.parquet", KG_OUT / "validation_33_model_ready.parquet"]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required input:\n{p}")

    train130 = pd.read_parquet(KG_OUT / "train_130_model_ready.parquet").copy()
    val33 = pd.read_parquet(KG_OUT / "validation_33_model_ready.parquet").copy()
    lrrk2 = pd.read_parquet(LRRK2_7).copy()
    e46k = pd.read_parquet(E46K_EXCLUDED).copy()
    manifest = pd.read_csv(KG_OUT / "protein_KG_manifest_1011.csv")
    proteins = manifest["protein"].astype(str).tolist()
    scaler: StandardScaler = joblib.load(KG_OUT / "training_rank_scaler.joblib")

    def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
        lower = {str(c).lower(): c for c in df.columns}
        for x in candidates:
            if x.lower() in lower:
                return lower[x.lower()]
        for c in df.columns:
            for x in candidates:
                if x.lower() in str(c).lower():
                    return c
        return None

    sample_col = find_col(lrrk2, ["sample_id", "sample", "participant_id", "subject_id", "id"])
    center_col = find_col(lrrk2, ["center", "centre", "site"])
    assert sample_col is not None
    assert len(lrrk2) == 7, f"Expected exactly seven LRRK2 participants, found {len(lrrk2)}"
    assert len(e46k) == 1, "The permanently excluded internal E46K audit file must contain one row."

    lrrk2[sample_col] = lrrk2[sample_col].astype(str).str.strip()
    e46k_sample_col = find_col(e46k, ["sample_id", "sample", "participant_id", "subject_id", "id"])
    assert e46k_sample_col is not None
    e46k_ids = set(e46k[e46k_sample_col].astype(str).str.strip())
    assert set(lrrk2[sample_col]).isdisjoint(e46k_ids)
    assert set(lrrk2[sample_col]).isdisjoint(set(train130["sample_id"].astype(str)))
    assert set(lrrk2[sample_col]).isdisjoint(set(val33["sample_id"].astype(str)))

    missing_proteins = [p for p in proteins if p not in lrrk2.columns]
    assert not missing_proteins, f"LRRK2 matrix is missing proteins: {missing_proteins[:10]}"

    raw = lrrk2[proteins].apply(pd.to_numeric, errors="coerce")
    ranked = raw.rank(axis=1, method="average", pct=True, na_option="keep").to_numpy(np.float64)
    missing_mask = ~np.isfinite(ranked)
    training_reference = np.broadcast_to(scaler.mean_, ranked.shape)
    ranked[missing_mask] = training_reference[missing_mask]
    ready = scaler.transform(ranked).astype(np.float32)
    assert np.isfinite(ready).all()

    lrrk2_ready = pd.DataFrame(ready, columns=proteins)
    lrrk2_ready.insert(0, "sample_id", lrrk2[sample_col].values)
    if center_col is not None:
        centers = lrrk2[center_col].astype(str).str.strip().values
    else:
        centers = np.repeat("DONOSTIA", 7)
    lrrk2_ready.insert(1, "center", centers)
    lrrk2_ready["group"] = "PD_LRRK2"
    lrrk2_ready["label"] = 1
    lrrk2_ready["domain_label"] = lrrk2_ready["center"].str.upper().map(
        {"CRUCES": 0, "DONOSTIA": 1}
    )
    assert lrrk2_ready["domain_label"].notna().all()
    lrrk2_ready["domain_label"] = lrrk2_ready["domain_label"].astype(int)
    lrrk2_ready["PD_spectrum_group"] = "LRRK2-PD"

    train_spectrum = train130.copy()
    train_spectrum["PD_spectrum_group"] = np.where(
        train_spectrum["label"].astype(int) == 0, "Control", "iPD"
    )

    # Align columns without permitting the excluded E46K frame to enter concatenation.
    for c in train_spectrum.columns:
        if c not in lrrk2_ready.columns:
            lrrk2_ready[c] = np.nan
    for c in lrrk2_ready.columns:
        if c not in train_spectrum.columns:
            train_spectrum[c] = np.nan
    lrrk2_ready = lrrk2_ready[train_spectrum.columns]

    train137 = pd.concat([train_spectrum, lrrk2_ready], ignore_index=True)

    assert len(train137) == 137
    assert train137["sample_id"].astype(str).nunique() == 137
    assert len(val33) == 33
    assert set(train137["sample_id"].astype(str)).isdisjoint(set(val33["sample_id"].astype(str)))
    assert set(train137["sample_id"].astype(str)).isdisjoint(e46k_ids)
    assert (train137["label"] == 0).sum() == 70
    assert (train137["label"] == 1).sum() == 67
    assert (train137["PD_spectrum_group"] == "iPD").sum() == 60
    assert (train137["PD_spectrum_group"] == "LRRK2-PD").sum() == 7
    assert not train137["PD_spectrum_group"].astype(str).str.contains("E46K", case=False).any()

    lrrk2_ready.to_parquet(LORA_DATA_OUT / "PXD068184_LRRK2_7_model_ready.parquet", index=False)
    train137.to_parquet(LORA_DATA_OUT / "PXD068184_PD_spectrum_LoRA_train_137.parquet", index=False)
    val33.to_parquet(LORA_DATA_OUT / "PXD068184_internal_validation_33_untouched.parquet", index=False)

    manifest_cols = [
        c for c in ["sample_id", "center", "group", "PD_spectrum_group", "label", "domain_label"]
        if c in train137.columns
    ]
    train137[manifest_cols].to_csv(LORA_DATA_OUT / "PD_spectrum_training_manifest_137.csv", index=False)

    audit = {
        "original_training_n": 130,
        "LRRK2_added": 7,
        "internal_E46K_added": 0,
        "internal_E46K_permanently_excluded": 1,
        "final_PD_spectrum_training_n": 137,
        "train_control": 70,
        "train_iPD": 60,
        "train_LRRK2": 7,
        "train_PD_spectrum": 67,
        "validation_n": 33,
        "genetic_PD_in_validation": 0,
        "protein_features": 1011,
        "new_scaler_fitted": False,
        "PD_spectrum_definition": "iPD + LRRK2-PD",
    }
    with open(LORA_DATA_OUT / "PD_spectrum_137_cohort_audit.json", "w") as f:
        json.dump(audit, f, indent=2)

    return train137, val33, manifest


# =============================================================================
# 8. PD-SPECTRUM LoRA TRAINING ON FROZEN V2
# =============================================================================
def train_lora_137(
    train: pd.DataFrame,
    val: pd.DataFrame,
    manifest: pd.DataFrame,
    device: torch.device,
) -> None:
    proteins = manifest["protein"].astype(str).tolist()
    Xtr = torch.tensor(train[proteins].to_numpy(np.float32), dtype=torch.float32)
    ytr = torch.tensor(train["label"].to_numpy(np.float32), dtype=torch.float32)
    Xva = torch.tensor(val[proteins].to_numpy(np.float32), dtype=torch.float32, device=device)
    y_val = val["label"].to_numpy(int)

    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=16,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    prior = torch.tensor(manifest["PD_KG_prior"].to_numpy(np.float32), dtype=torch.float32)
    model = KGGatedDomainModel(1011, prior)
    model.load_state_dict(torch.load(V2_OUT / "best_KG_DA_V2.pt", map_location="cpu"), strict=True)
    for p in model.parameters():
        p.requires_grad = False
    model = add_paper_lora(model).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]

    assert total_params == 151_408, f"Expected 151,408 total parameters, found {total_params:,}"
    assert trainable_params == 10_010, f"Expected 10,010 LoRA parameters, found {trainable_params:,}"
    assert frozen_params == 141_398
    assert all(name.endswith(".A") or name.endswith(".B") for name in trainable_names)

    criterion = nn.BCEWithLogitsLoss()  # 70 vs 67: no class weighting/resampling.
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=1e-4,
    )

    MAX_EPOCHS, MIN_EPOCHS, PATIENCE = 120, 25, 25

    def evaluate() -> tuple[float, np.ndarray, dict[str, float]]:
        model.eval()
        with torch.no_grad():
            logits, _, emb = model(Xva, alpha=0.0)
            prob = torch.sigmoid(logits).cpu().numpy()
        return float(criterion(logits, torch.tensor(y_val, dtype=torch.float32, device=device)).item()), emb, binary_metrics(y_val, prob, 0.5) | {"prob": prob}

    # Epoch 0 = exact frozen V2 because every LoRA B matrix starts at zero.
    val_loss0, _, met0 = evaluate()
    best_auc = float(met0["AUROC"])
    best_auprc = float(met0["AUPRC"])
    best_epoch = 0
    wait = 0
    torch.save(model.state_dict(), LORA_TRAIN_OUT / "best_PD_spectrum_LoRA_v2_137.pt")

    history: list[dict] = [
        {
            "epoch": 0,
            "train_loss": np.nan,
            "validation_loss": val_loss0,
            "train_AUROC": np.nan,
            "train_AUPRC": np.nan,
            "validation_AUROC": best_auc,
            "validation_AUPRC": best_auprc,
            "validation_accuracy": met0["Accuracy"],
            "validation_F1": met0["F1"],
            "learning_rate": 1e-4,
        }
    ]

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses, train_prob, train_true = [], [], []

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, _, _ = model(xb, alpha=0.0)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            losses.append(loss.item())
            train_prob.extend(torch.sigmoid(logits).detach().cpu().numpy())
            train_true.extend(yb.detach().cpu().numpy())

        train_prob = np.asarray(train_prob)
        train_true = np.asarray(train_true).astype(int)
        train_auc = float(roc_auc_score(train_true, train_prob))
        train_auprc = float(average_precision_score(train_true, train_prob))

        val_loss, _, val_met = evaluate()
        val_prob = np.asarray(val_met.pop("prob"))
        val_auc, val_auprc = float(val_met["AUROC"]), float(val_met["AUPRC"])

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_loss": val_loss,
                "train_AUROC": train_auc,
                "train_AUPRC": train_auprc,
                "validation_AUROC": val_auc,
                "validation_AUPRC": val_auprc,
                "validation_accuracy": val_met["Accuracy"],
                "validation_F1": val_met["F1"],
                "validation_precision": val_met["Precision"],
                "validation_recall": val_met["Recall"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        improved = (
            val_auc > best_auc + 1e-4
            or (abs(val_auc - best_auc) <= 1e-4 and val_auprc > best_auprc + 1e-4)
        )
        if improved:
            best_auc, best_auprc, best_epoch, wait = val_auc, val_auprc, epoch, 0
            torch.save(model.state_dict(), LORA_TRAIN_OUT / "best_PD_spectrum_LoRA_v2_137.pt")
        else:
            wait += 1

        if epoch == 1 or epoch % 10 == 0 or improved:
            print(
                f"LoRA {epoch:03d} | train loss {np.mean(losses):.4f} | "
                f"val AUROC {val_auc:.4f} | AUPRC {val_auprc:.4f} | F1 {val_met['F1']:.4f}"
            )
        if epoch >= MIN_EPOCHS and wait >= PATIENCE:
            print(f"LoRA early stopping at epoch {epoch}")
            break

    pd.DataFrame(history).to_csv(LORA_TRAIN_OUT / "PD_spectrum_LoRA_137_training_history.csv", index=False)

    # 137-person checkpoint and lock the threshold from internal validation only.
    model.load_state_dict(
        torch.load(LORA_TRAIN_OUT / "best_PD_spectrum_LoRA_v2_137.pt", map_location=device),
        strict=True,
    )
    model.eval()
    with torch.no_grad():
        logits, _, embeddings = model(Xva, alpha=0.0)
        probability = torch.sigmoid(logits).cpu().numpy()

    fpr, tpr, thresholds = roc_curve(y_val, probability)
    locked_threshold = float(thresholds[np.argmax(tpr - fpr)])
    final_metrics = binary_metrics(y_val, probability, locked_threshold)

    pred = val[["sample_id", "center", "group", "label", "domain_label"]].copy()
    pred["probability_PD_spectrum"] = probability
    pred["locked_threshold"] = locked_threshold
    pred["prediction"] = (probability >= locked_threshold).astype(int)
    pred.to_csv(LORA_TRAIN_OUT / "PD_spectrum_LoRA_137_internal_validation_predictions.csv", index=False)

    emb = pd.DataFrame(
        embeddings.cpu().numpy(), columns=[f"embedding_{i+1}" for i in range(64)]
    )
    emb.insert(0, "sample_id", val["sample_id"].values)
    emb.to_parquet(LORA_TRAIN_OUT / "PD_spectrum_LoRA_137_validation_embeddings.parquet", index=False)

    config = {
        "task": "Control vs PD-spectrum",
        "PD_spectrum_development_definition": ["iPD", "LRRK2-PD"],
        "internal_E46K_used": False,
        "internal_E46K_permanently_excluded_n": 1,
        "base_model": "Frozen PrimeKG-guided + CRUCES/DONOSTIA domain-adaptive V2",
        "fresh_LoRA_from_V2": True,
        "train_n": 137,
        "train_control": 70,
        "train_PD_spectrum": 67,
        "train_iPD": 60,
        "train_LRRK2": 7,
        "train_E46K": 0,
        "validation_n": 33,
        "validation_control": 17,
        "validation_iPD": 16,
        "genetic_PD_in_validation": 0,
        "protein_features": 1011,
        "total_parameters": int(total_params),
        "frozen_parameters": int(frozen_params),
        "LoRA_trainable_parameters": int(trainable_params),
        "trainable_fraction_percent": float(100 * trainable_params / total_params),
        "LoRA_layers": {
            "encoder_0": {"rank": 8, "alpha": 16, "dropout": 0.10},
            "encoder_4": {"rank": 4, "alpha": 8, "dropout": 0.10},
            "disease_head": {"rank": 2, "alpha": 4, "dropout": 0.05},
        },
        "optimizer": "AdamW",
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "batch_size": 16,
        "max_epochs": MAX_EPOCHS,
        "checkpoint_primary_metric": "validation AUROC",
        "checkpoint_tie_breaker": "validation AUPRC",
        "best_epoch": int(best_epoch),
        "best_validation_AUROC": float(best_auc),
        "best_validation_AUPRC": float(best_auprc),
        "locked_internal_threshold": locked_threshold,
        "external_dataset_used": False,
        "external_threshold_optimization": False,
    }
    with open(LORA_TRAIN_OUT / "PD_spectrum_LoRA_137_training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    pd.DataFrame([final_metrics]).to_csv(
        LORA_TRAIN_OUT / "PD_spectrum_LoRA_137_internal_threshold_metrics.csv", index=False
    )

    print("\n" + "=" * 88)
    print("PAPER-ALIGNED LoRA TRAINING COMPLETE")
    print("=" * 88)
    print("Training: 137 = 70 control + 60 iPD + 7 LRRK2-PD")
    print("Internal E46K used: 0")
    print(f"Parameters: {trainable_params:,} trainable / {total_params:,} total")
    print(f"Best epoch: {best_epoch}")
    print(f"Locked internal threshold: {locked_threshold:.4f}")
    print(f"Outputs: {LORA_TRAIN_OUT}")


# =============================================================================
# 9. MAIN
# =============================================================================
def main() -> None:
    mount_drive_if_colab()
    set_reproducibility()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_raw, val_raw, proteins = make_fixed_split()
    train_ready, val_ready, manifest = prepare_kg_inputs(train_raw, val_raw, proteins)
    train_v2(train_ready, val_ready, manifest, device)
    train137, val33, manifest = build_lora_137_cohort()
    train_lora_137(train137, val33, manifest, device)


if __name__ == "__main__":
    main()
