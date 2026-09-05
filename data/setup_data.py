"""
Dataset setup and verification helper for Credit Card Fraud Detection.

Supports:
1. Verifying presence and schema of real 'creditcard.csv'.
2. Downloading from Kaggle if credentials or kagglehub are configured.
3. Generating a realistic schema-compliant synthetic creditcard dataset
   (Time, V1-V28, Amount, Class) for instant local testing and CI/CD.
"""
import os
import sys
import csv
import random
import argparse
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = DATA_DIR / "creditcard.csv"

# Standard schema for Kaggle credit card fraud detection dataset
EXPECTED_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]


def check_dataset_status(csv_path: Path = DEFAULT_CSV_PATH) -> dict:
    """Check if the dataset exists and summarize its records."""
    if not csv_path.exists():
        return {"exists": False, "path": str(csv_path), "rows": 0, "fraud_count": 0}

    total_rows = 0
    fraud_count = 0
    header_valid = False

    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header and all(col in header for col in ["Time", "Amount", "Class"]):
            header_valid = True
            class_idx = header.index("Class")
            for row in reader:
                total_rows += 1
                try:
                    if int(float(row[class_idx])) == 1:
                        fraud_count += 1
                except (ValueError, IndexError):
                    pass

    return {
        "exists": True,
        "path": str(csv_path),
        "valid_header": header_valid,
        "rows": total_rows,
        "fraud_count": fraud_count,
        "fraud_ratio": (fraud_count / total_rows) if total_rows > 0 else 0.0,
    }


def generate_synthetic_dataset(
    output_path: Path = DEFAULT_CSV_PATH,
    num_rows: int = 5000,
    fraud_rate: float = 0.005,
    seed: int = 42,
) -> None:
    """
    Generate a schema-compliant synthetic dataset matching Kaggle's creditcard.csv.
    This enables immediate development and module testing without requiring
    manual 150MB file downloads, while matching exact column names and statistical behaviors.
    """
    random.seed(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic creditcard dataset ({num_rows} transactions)...")
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(EXPECTED_COLUMNS)

        current_time = 0.0
        for i in range(num_rows):
            # Advance time in increments (seconds)
            current_time += random.uniform(0.5, 30.0)
            is_fraud = 1 if random.random() < fraud_rate else 0

            # V1-V28 PCA features: standard normal with distinct shifts on fraud cases
            v_features = []
            for v_idx in range(1, 29):
                if is_fraud:
                    # Key fraud indicators in real creditcard dataset typically show in V1, V4, V10, V12, V14
                    if v_idx in [4, 11]:
                        val = random.gauss(3.5, 1.5)
                    elif v_idx in [10, 12, 14, 17]:
                        val = random.gauss(-4.0, 1.8)
                    else:
                        val = random.gauss(0.0, 1.5)
                else:
                    val = random.gauss(0.0, 1.0)
                v_features.append(f"{val:.6f}")

            # Amount: lognormal distribution (normal transactions mostly < $100, occasional spikes)
            if is_fraud:
                amount = random.choice([
                    random.uniform(1.0, 15.0),    # Small card testing probe
                    random.uniform(250.0, 1800.0) # High-value extraction
                ])
            else:
                amount = max(0.5, random.lognormvariate(3.0, 1.2))

            row = [f"{current_time:.1f}"] + v_features + [f"{amount:.2f}", is_fraud]
            writer.writerow(row)

    print(f"Successfully generated {num_rows} transactions -> {output_path}")


def download_kaggle_dataset(output_path: Path = DEFAULT_CSV_PATH) -> bool:
    """Attempt to download from Kaggle using kagglehub or kaggle CLI."""
    try:
        import kagglehub
        print("Downloading dataset via kagglehub...")
        path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        downloaded_file = Path(path) / "creditcard.csv"
        if downloaded_file.exists():
            import shutil
            shutil.copyfile(downloaded_file, output_path)
            print(f"Copied dataset to {output_path}")
            return True
    except Exception as e:
        print(f"kagglehub download unavailable: {e}")

    print("\nTo use the original Kaggle dataset manually:")
    print("1. Visit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print(f"2. Download 'creditcard.csv' and place it at: {output_path.resolve()}")
    return False


def main():
    parser = argparse.ArgumentParser(description="Dataset setup and verification tool")
    parser.add_argument("--generate", action="store_true", help="Generate synthetic schema-compliant dataset")
    parser.add_argument("--download", action="store_true", help="Download from Kaggle")
    parser.add_argument("--rows", type=int, default=10000, help="Number of rows for synthetic generation")
    parser.add_argument("--fraud-rate", type=float, default=0.005, help="Fraud rate (e.g. 0.005 for 0.5%)")
    parser.add_argument("--path", type=str, default=str(DEFAULT_CSV_PATH), help="Custom target CSV path")
    args = parser.parse_args()

    target_path = Path(args.path)
    status = check_dataset_status(target_path)

    if args.download:
        success = download_kaggle_dataset(target_path)
        if success:
            return

    if args.generate or not status["exists"]:
        if not status["exists"] and not args.generate:
            print(f"Notice: '{target_path.name}' not found. Generating sample data for instant execution.")
        generate_synthetic_dataset(target_path, num_rows=args.rows, fraud_rate=args.fraud_rate)
        status = check_dataset_status(target_path)

    print("\n--- Dataset Status ---")
    print(f"Path:        {status['path']}")
    print(f"Exists:      {status['exists']}")
    print(f"Total Rows:  {status['rows']:,}")
    print(f"Fraud Rows:  {status['fraud_count']:,} ({status['fraud_ratio']*100:.3f}%)")


if __name__ == "__main__":
    main()
