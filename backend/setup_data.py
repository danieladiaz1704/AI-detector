"""
backend/setup_data.py

Run this ONCE to download the dataset from Kaggle.
Reads credentials from backend/.env — never hardcode your key.

Usage:
    cd backend
    python setup_data.py
"""

import os
import sys
import subprocess


BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
ENV_PATH  = os.path.join(BASE_DIR, ".env")

DATASET_SLUG    = "shanegerami/ai-vs-human-text"
EXPECTED_FILE   = os.path.join(DATA_DIR, "dataset.csv")
DOWNLOADED_NAME = "AI_Human.csv"


# ─────────────────────────────────────────────────────────────
def load_env(path: str) -> None:
    """Read KEY=VALUE lines from .env and set as env variables.
    Strips surrounding quotes and trailing commas from values.
    """
    if not os.path.exists(path):
        print(f"ERROR: .env file not found at {path}")
        print("Create backend/.env with:")
        print("  KAGGLE_USERNAME=yourname")
        print("  KAGGLE_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        sys.exit(1)

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            # Strip whitespace, quotes, and trailing commas
            value = value.strip().strip('"').strip("'").rstrip(",").strip()
            os.environ[key.strip()] = value

    print(f"Loaded credentials from {path}")


def check_credentials() -> None:
    username = os.environ.get("KAGGLE_USERNAME", "")
    key      = os.environ.get("KAGGLE_KEY", "")

    if not username or not key:
        print("ERROR: KAGGLE_USERNAME or KAGGLE_KEY missing from .env")
        sys.exit(1)

    print(f"Kaggle credentials found for user: {username}")


def download_dataset() -> None:
    import shutil

    os.makedirs(DATA_DIR, exist_ok=True)

    # Find the kaggle executable (works on Windows, Mac, Linux)
    kaggle_exe = shutil.which("kaggle")

    if not kaggle_exe:
        # Windows fallback: look in Scripts/ next to the current Python
        scripts_dir = os.path.join(os.path.dirname(sys.executable), "Scripts")
        candidate   = os.path.join(scripts_dir, "kaggle.exe")
        if os.path.exists(candidate):
            kaggle_exe = candidate

    if not kaggle_exe:
        print("ERROR: kaggle executable not found.")
        print("Run:  pip install kaggle")
        sys.exit(1)

    print(f"Using kaggle at: {kaggle_exe}")
    print(f"Downloading dataset: {DATASET_SLUG}")
    print(f"Saving to: {DATA_DIR}")
    print("This may take a few minutes (~500 MB)...\n")

    result = subprocess.run(
        [kaggle_exe, "datasets", "download",
         "-d", DATASET_SLUG,
         "-p", DATA_DIR,
         "--unzip"],
        env=os.environ.copy(),
    )

    if result.returncode != 0:
        print("\nERROR: Kaggle download failed.")
        print("Double-check your KAGGLE_USERNAME and KAGGLE_KEY in .env")
        sys.exit(1)

    print("\nDownload complete.")


def rename_to_standard() -> None:
    """Rename the downloaded file to dataset.csv if needed."""
    if os.path.exists(EXPECTED_FILE):
        print(f"dataset.csv already exists — skipping rename.")
        return

    downloaded = os.path.join(DATA_DIR, DOWNLOADED_NAME)
    if os.path.exists(downloaded):
        os.rename(downloaded, EXPECTED_FILE)
        print(f"Renamed {DOWNLOADED_NAME} → dataset.csv")
        return

    # Check what CSV files are actually in the folder
    files     = os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else []
    csv_files = [f for f in files if f.endswith(".csv")]

    if len(csv_files) == 1:
        src = os.path.join(DATA_DIR, csv_files[0])
        os.rename(src, EXPECTED_FILE)
        print(f"Renamed {csv_files[0]} → dataset.csv")
    elif len(csv_files) > 1:
        print("Multiple CSV files found. Rename the correct one manually:")
        for f in csv_files:
            print(f"  {f}")
        sys.exit(1)
    else:
        print(f"ERROR: No CSV file found in {DATA_DIR}")
        print(f"Files present: {files}")
        sys.exit(1)


def verify_dataset() -> None:
    try:
        import pandas as pd
    except ImportError:
        print("pandas not installed — skipping verification.")
        return

    print("Verifying dataset...")
    df = pd.read_csv(EXPECTED_FILE, nrows=5)

    if "text" not in df.columns or "generated" not in df.columns:
        print(f"ERROR: Expected columns 'text' and 'generated'.")
        print(f"Found columns: {df.columns.tolist()}")
        sys.exit(1)

    total = sum(1 for _ in open(EXPECTED_FILE, encoding="utf-8", errors="ignore")) - 1
    print(f"Columns   : {df.columns.tolist()}")
    print(f"Total rows: {total:,}")
    print("Dataset is ready.")


# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  AI Text Detector — Dataset Setup")
    print("=" * 50)

    if os.path.exists(EXPECTED_FILE):
        print(f"dataset.csv already exists — skipping download.")
        verify_dataset()
        return

    load_env(ENV_PATH)
    check_credentials()
    download_dataset()
    rename_to_standard()
    verify_dataset()

    print("\nSetup complete. You can now run the training scripts.")


if __name__ == "__main__":
    main()