
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Optional packages used only when DOWNLOAD_DATA=True.
try:
    import requests
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - audit can run without network helpers.
    requests = None
    tqdm = None


# =============================================================================
# 0. CONFIGURATION
# =============================================================================
SEED = 2026
ROOT = Path("/content/drive/MyDrive/Parkinson_Tear_Proteomics")
RUN = ROOT / "05_PAPER_ALIGNED_137"

RAW_INTERNAL = ROOT / "01_data/raw/PXD068184"
RAW_EXTERNAL = ROOT / "01_data/raw/PXD028811"
LEGACY_SPLIT = ROOT / "01_data/splits/PXD068184"
EXTERNAL_PROCESSED = ROOT / "01_data/external_processed/PXD028811"

DATASET_OUT = RUN / "01_dataset"
AUDIT_OUT = DATASET_OUT / "audit"
MANIFEST_OUT = DATASET_OUT / "manifests"
DOWNLOAD_OUT = DATASET_OUT / "pride_metadata"
for p in [AUDIT_OUT, MANIFEST_OUT, DOWNLOAD_OUT]:
    p.mkdir(parents=True, exist_ok=True)

# Set True only when you intentionally want network downloads in Colab.
DOWNLOAD_DATA = False

# Canonical internal search-engine output file used by the supplied pipeline.
INTERNAL_XLSX = RAW_INTERNAL / "230417_230214C_Acera_Lagrima_DIANN.xlsx"

# Existing canonical matrices/assets created by the original notebook.
# The cleaned scripts validate and reuse them rather than silently inventing a
# different DIA-NN/PEAKS conversion from raw mass-spectrometry files.
INTERNAL_163_MATRIX = LEGACY_SPLIT / "binary_163_before_split.parquet"
GENETIC_8_MATRIX = LEGACY_SPLIT / "excluded_genetic_patients.parquet"

EXPECTED_INTERNAL_COUNTS = {
    "control": 87,
    "iPD": 76,
    "PD (LRRK2)": 7,
    "PD(E46K)": 1,
}


# =============================================================================
# 1. COLAB / DRIVE SETUP
# =============================================================================
def mount_drive_if_colab() -> None:
    """Mount Google Drive when executed inside Google Colab."""
    try:
        from google.colab import drive  # type: ignore
    except Exception:
        return
    drive.mount("/content/drive", force_remount=False)


# =============================================================================
# 2. OPTIONAL OFFICIAL PRIDE FILE DISCOVERY / DOWNLOAD
# =============================================================================
def _extract_file_records(payload: object) -> list[dict]:
    """Extract PRIDE file records across common v2/v3 response layouts."""
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        return []

    for key in ("files", "content"):
        value = payload.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]

    embedded = payload.get("_embedded")
    if isinstance(embedded, dict):
        for key in ("files", "projectFiles"):
            value = embedded.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
    return []


def pride_file_list(accession: str) -> pd.DataFrame:
    """Query official PRIDE metadata and return a compact project file list."""
    if requests is None:
        raise RuntimeError("requests is required for PRIDE network access.")

    endpoints = [
        f"https://www.ebi.ac.uk/pride/ws/archive/v3/projects/{accession}/files?pageSize=1000",
        f"https://www.ebi.ac.uk/pride/ws/archive/v2/projects/{accession}/files?pageSize=1000",
    ]
    last_error: Exception | None = None
    records: list[dict] = []

    for url in endpoints:
        try:
            response = requests.get(url, timeout=45, headers={"Accept": "application/json"})
            if not response.ok:
                continue
            records = _extract_file_records(response.json())
            if records:
                break
        except Exception as exc:  # pragma: no cover - network dependent.
            last_error = exc

    if not records:
        raise RuntimeError(
            f"Could not retrieve a PRIDE file list for {accession}. Last error: {last_error}"
        )

    rows = []
    for rec in records:
        name = rec.get("fileName") or rec.get("name") or rec.get("filename")
        size = rec.get("fileSizeBytes") or rec.get("fileSize") or rec.get("size")
        category = rec.get("fileCategory")
        if isinstance(category, dict):
            category = category.get("value") or category.get("name")

        locations = rec.get("publicFileLocations") or rec.get("downloadLinks") or []
        if isinstance(locations, str):
            locations = [locations]
        if isinstance(locations, dict):
            locations = list(locations.values())

        urls: list[str] = []
        if isinstance(locations, list):
            for loc in locations:
                if isinstance(loc, str):
                    urls.append(loc)
                elif isinstance(loc, dict):
                    for key in ("value", "url", "href"):
                        if isinstance(loc.get(key), str):
                            urls.append(loc[key])
                            break

        links = rec.get("_links")
        if isinstance(links, dict):
            for value in links.values():
                if isinstance(value, dict) and isinstance(value.get("href"), str):
                    href = value["href"]
                    if href.startswith(("http://", "https://", "ftp://")):
                        urls.append(href)

        rows.append(
            {
                "accession": accession,
                "file_name": str(name) if name is not None else "",
                "category": category,
                "size_bytes": pd.to_numeric(size, errors="coerce"),
                "download_url": next(
                    (u for u in urls if str(u).startswith(("https://", "http://"))),
                    next((u for u in urls if str(u).startswith("ftp://")), ""),
                ),
            }
        )

    frame = pd.DataFrame(rows).drop_duplicates("file_name").reset_index(drop=True)
    frame.to_csv(DOWNLOAD_OUT / f"{accession}_official_file_list.csv", index=False)
    return frame


def download_file(url: str, destination: Path, chunk_mb: int = 4) -> Path:
    """Streaming download with a progress bar; existing non-empty files are reused."""
    if requests is None:
        raise RuntimeError("requests is required for downloads.")
    # PRIDE metadata can expose ftp:// links; the same archive is available over HTTPS.
    if url.startswith("ftp://ftp.pride.ebi.ac.uk/"):
        url = "https://ftp.pride.ebi.ac.uk/" + url.split("ftp://ftp.pride.ebi.ac.uk/", 1)[1]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        bar = (
            tqdm(total=total, unit="B", unit_scale=True, desc=destination.name)
            if tqdm is not None
            else None
        )
        with open(destination, "wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_mb * 1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                if bar is not None:
                    bar.update(len(chunk))
        if bar is not None:
            bar.close()
    return destination


def download_pride_selection(
    accession: str,
    destination: Path,
    file_name_patterns: Iterable[str],
) -> list[Path]:
    """Download only explicitly selected PRIDE project files, never all raw MS files."""
    listing = pride_file_list(accession)
    regexes = [re.compile(p, flags=re.I) for p in file_name_patterns]
    selected = listing[
        listing["file_name"].map(lambda x: any(r.search(str(x)) for r in regexes))
    ].copy()

    if selected.empty:
        raise RuntimeError(f"No PRIDE files matched the requested patterns for {accession}.")

    downloaded: list[Path] = []
    for row in selected.itertuples(index=False):
        if not row.download_url:
            raise RuntimeError(f"No public download URL exposed for {row.file_name}")
        downloaded.append(download_file(str(row.download_url), destination / str(row.file_name)))
    return downloaded


# =============================================================================
# 3. PXD068184 PARTICIPANT AUDIT
# =============================================================================
def read_internal_metadata() -> pd.DataFrame:
    if not INTERNAL_XLSX.exists():
        raise FileNotFoundError(
            f"Missing PXD068184 workbook:\n{INTERNAL_XLSX}\n"
            "Set DOWNLOAD_DATA=True or place the official workbook at this path."
        )

    samples = pd.read_excel(INTERNAL_XLSX, sheet_name="Samples")
    required = ["Name", "Group1", "Group2"]
    missing = [c for c in required if c not in samples.columns]
    if missing:
        raise RuntimeError(f"PXD068184 Samples sheet is missing columns: {missing}")

    meta = samples[required].rename(
        columns={"Name": "sample_id", "Group1": "center", "Group2": "group"}
    ).copy()
    for c in meta.columns:
        meta[c] = meta[c].astype(str).str.strip()

    assert len(meta) == 171, f"Expected 171 PXD068184 participants, found {len(meta)}"
    assert meta["sample_id"].is_unique, "Duplicate PXD068184 participant IDs detected."

    actual = meta["group"].value_counts().to_dict()
    for group, expected in EXPECTED_INTERNAL_COUNTS.items():
        assert actual.get(group, 0) == expected, (
            f"{group}: expected {expected}, found {actual.get(group, 0)}"
        )
    return meta


def create_paper_aligned_internal_manifests(meta: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create the exact paper-aligned 163 + 7 + permanently-excluded-1 grouping."""
    development = meta[meta["group"].isin(["control", "iPD"])].copy()
    development["label"] = development["group"].map({"control": 0, "iPD": 1}).astype(int)

    lrrk2 = meta[meta["group"].eq("PD (LRRK2)")].copy()
    e46k = meta[meta["group"].eq("PD(E46K)")].copy()

    assert len(development) == 163
    assert (development["group"] == "control").sum() == 87
    assert (development["group"] == "iPD").sum() == 76
    assert len(lrrk2) == 7
    assert len(e46k) == 1

    # This is the critical fix requested by the user and required by the paper.
    assert set(e46k["sample_id"]).isdisjoint(set(development["sample_id"]))
    assert set(e46k["sample_id"]).isdisjoint(set(lrrk2["sample_id"]))

    development.to_csv(
        MANIFEST_OUT / "PXD068184_development_163_control_iPD.csv", index=False
    )
    lrrk2.to_csv(
        MANIFEST_OUT / "PXD068184_LRRK2_7_held_for_LoRA.csv", index=False
    )
    e46k.to_csv(
        MANIFEST_OUT / "PXD068184_E46K_1_PERMANENTLY_EXCLUDED.csv", index=False
    )

    return {"development": development, "lrrk2": lrrk2, "e46k": e46k}


def verify_canonical_internal_matrices(parts: dict[str, pd.DataFrame]) -> None:
    """Verify the supplied notebook's canonical 1,011-protein matrices against manifests."""
    development = parts["development"]
    lrrk2 = parts["lrrk2"]
    e46k = parts["e46k"]

    if not INTERNAL_163_MATRIX.exists():
        raise FileNotFoundError(
            f"Canonical 163-person proteomic matrix not found:\n{INTERNAL_163_MATRIX}\n"
            "The supplied notebook used this prepared DIA-NN matrix; this cleaned code "
            "does not silently substitute a different raw-MS processing workflow."
        )

    dev_matrix = pd.read_parquet(INTERNAL_163_MATRIX)
    assert len(dev_matrix) == 163
    assert "sample_id" in dev_matrix.columns
    assert dev_matrix["sample_id"].astype(str).is_unique
    assert set(dev_matrix["sample_id"].astype(str)) == set(development["sample_id"].astype(str))

    meta_cols = {"sample_id", "center", "group", "label", "split", "domain_label"}
    protein_cols = [
        c for c in dev_matrix.columns
        if c not in meta_cols and pd.api.types.is_numeric_dtype(dev_matrix[c])
    ]
    assert len(protein_cols) == 1011, (
        f"Expected exactly 1,011 numeric protein features, found {len(protein_cols)}"
    )

    if not GENETIC_8_MATRIX.exists():
        raise FileNotFoundError(
            f"Canonical genetic participant matrix not found:\n{GENETIC_8_MATRIX}"
        )

    genetic = pd.read_parquet(GENETIC_8_MATRIX).copy()
    sample_col = next(
        (c for c in ["sample_id", "sample", "participant_id", "subject_id", "id"] if c in genetic.columns),
        None,
    )
    group_col = next(
        (c for c in ["group", "disease", "diagnosis", "condition", "class"] if c in genetic.columns),
        None,
    )
    assert sample_col is not None and group_col is not None
    genetic[sample_col] = genetic[sample_col].astype(str).str.strip()

    lrrk2_ids = set(lrrk2["sample_id"].astype(str))
    e46k_ids = set(e46k["sample_id"].astype(str))
    assert lrrk2_ids.issubset(set(genetic[sample_col]))
    assert e46k_ids.issubset(set(genetic[sample_col]))

    # Save only the seven allowed genetic cases for the methodology file.
    genetic_lrrk2 = genetic[genetic[sample_col].isin(lrrk2_ids)].copy()
    genetic_e46k = genetic[genetic[sample_col].isin(e46k_ids)].copy()
    assert len(genetic_lrrk2) == 7
    assert len(genetic_e46k) == 1

    genetic_lrrk2.to_parquet(
        MANIFEST_OUT / "PXD068184_LRRK2_7_with_proteins.parquet", index=False
    )
    genetic_e46k.to_parquet(
        MANIFEST_OUT / "PXD068184_E46K_1_EXCLUDED_with_proteins.parquet", index=False
    )

    audit = {
        "internal_original_n": 171,
        "development_control_iPD_n": 163,
        "development_control_n": 87,
        "development_iPD_n": 76,
        "LRRK2_held_for_LoRA_n": 7,
        "internal_E46K_permanently_excluded_n": 1,
        "protein_features": 1011,
        "paper_aligned_final_LoRA_train_n_expected": 137,
        "paper_aligned_final_LoRA_control_n_expected": 70,
        "paper_aligned_final_LoRA_PD_spectrum_n_expected": 67,
        "paper_aligned_final_LoRA_PD_spectrum_definition": "60 iPD + 7 LRRK2-PD",
    }
    with open(AUDIT_OUT / "PXD068184_paper_aligned_cohort_audit.json", "w") as f:
        json.dump(audit, f, indent=2)


# =============================================================================
# 4. PXD028811 EXTERNAL COHORT AUDIT
# =============================================================================
def audit_external_processed_matrix() -> None:
    """Audit the independent external cohort without using it for model development."""
    preferred = (
        EXTERNAL_PROCESSED
        / "model_ready"
        / "PXD028811_raw_area_internal_feature_space.parquet"
    )
    matrix = preferred
    if not matrix.exists():
        candidates = [
            p for p in EXTERNAL_PROCESSED.rglob("*.parquet")
            if any(k in p.name.lower() for k in ("raw_area", "internal_feature", "model_ready"))
        ] if EXTERNAL_PROCESSED.exists() else []
        if candidates:
            matrix = candidates[0]

    spec = {
        "dataset": "PXD028811",
        "role": "independent external validation only",
        "total_n": 54,
        "controls": 27,
        "PD_spectrum": 27,
        "PD_spectrum_composition": "24 iPD + 3 E46K-SNCA",
        "external_training": False,
        "external_LoRA_tuning": False,
        "external_domain_adaptation": False,
        "external_threshold_optimization": False,
        "external_feature_selection": False,
    }

    if matrix.exists():
        ext = pd.read_parquet(matrix)
        assert "sample_id" in ext.columns
        ext["sample_id"] = ext["sample_id"].astype(str).str.strip()
        ext = ext.drop_duplicates("sample_id")
        assert len(ext) == 54, f"Expected 54 external participants, found {len(ext)}"

        # This naming rule is the one used in the supplied final external code:
        # Ctrl* = control; all A*/Lag* PD-side samples = PD-spectrum.
        labels = np.where(
            ext["sample_id"].str.upper().str.startswith("CTRL"),
            "Control",
            "PD-spectrum",
        )
        counts = pd.Series(labels).value_counts().to_dict()
        assert counts.get("Control", 0) == 27
        assert counts.get("PD-spectrum", 0) == 27
        spec["processed_matrix"] = str(matrix)
        spec["processed_matrix_verified"] = True
    else:
        spec["processed_matrix"] = str(preferred)
        spec["processed_matrix_verified"] = False

    with open(AUDIT_OUT / "PXD028811_external_cohort_specification.json", "w") as f:
        json.dump(spec, f, indent=2)


# =============================================================================
# 5. MAIN
# =============================================================================
def main() -> None:
    mount_drive_if_colab()

    print("=" * 88)
    print("PAPER-ALIGNED DATASET ACQUISITION / AUDIT")
    print("=" * 88)
    print(f"Project root : {ROOT}")
    print(f"Clean run    : {RUN}")

    if DOWNLOAD_DATA:
        # PXD068184: download only the official DIA-NN workbook used by the model,
        
        try:
            download_pride_selection(
                "PXD068184",
                RAW_INTERNAL,
                [r"^230417_230214C_Acera_Lagrima_DIANN\.xlsx$"],
            )
        except Exception as exc:
            # Official PRIDE archive fallback for the exact model workbook.
            print(f"PRIDE API selection failed ({type(exc).__name__}); using archive URL.")
            download_file(
                "https://ftp.pride.ebi.ac.uk/pride/data/archive/2026/06/PXD068184/"
                "230417_230214C_Acera_Lagrima_DIANN.xlsx",
                INTERNAL_XLSX,
            )

        # PXD028811: capture the complete official file list. Raw-file selection is
        
        
        pride_file_list("PXD028811")

    meta = read_internal_metadata()
    parts = create_paper_aligned_internal_manifests(meta)
    verify_canonical_internal_matrices(parts)
    audit_external_processed_matrix()

    print("\nInternal PXD068184 paper-aligned cohort:")
    print("  control + iPD development : 163")
    print("  held LRRK2 for LoRA       : 7")
    print("  internal E46K excluded    : 1  [PERMANENT] ")
    print("  expected LoRA train       : 137 = 70 control + 60 iPD + 7 LRRK2")
    print("\nExternal PXD028811 remains:")
    print("  54 = 27 control + 24 iPD + 3 E46K-SNCA")
    print(f"\nOutputs: {DATASET_OUT}")


if __name__ == "__main__":
    main()
