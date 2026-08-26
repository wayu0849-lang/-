import os
import shutil
import argparse
from pathlib import Path
import kagglehub

DEFAULT_DATASET = "rajarshi2712/dogs-and-cats-classifier"
DEFAULT_TOKEN = os.getenv("KAGGLE_API_TOKEN", "KGAT_cfeccb4e9a16a465869fde84218388de")


def setup_kaggle_auth(token: str):
    """Authenticate with Kaggle using Kaggle API token."""
    if token:
        kagglehub.config.set_kaggle_api_token(token)
        print(f"[Auth] Kaggle API token configured successfully.")


def count_files_by_extension(directory: Path):
    """Count files grouped by extension in a directory recursively."""
    ext_counts = {}
    total = 0
    for p in directory.rglob("*"):
        if p.is_file():
            ext = p.suffix.lower() if p.suffix else "(no extension)"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
            total += 1
    return total, ext_counts


def download_and_setup_dataset(dataset_handle: str = DEFAULT_DATASET,
                              output_dir: str = "data",
                              token: str = DEFAULT_TOKEN,
                              copy_to_local: bool = True):
    """
    Download dataset via kagglehub and optionally copy to local project directory.
    """
    setup_kaggle_auth(token)

    print(f"\n[Download] Fetching dataset '{dataset_handle}' via kagglehub...")
    cached_path = Path(kagglehub.dataset_download(dataset_handle))
    print(f"[Success] Dataset downloaded to KaggleHub cache: {cached_path}")

    total_files, ext_counts = count_files_by_extension(cached_path)
    print(f"[Summary] Found {total_files} files in downloaded dataset:")
    for ext, count in ext_counts.items():
        print(f"  - {ext}: {count} files")

    if copy_to_local:
        dest_path = Path(output_dir)
        print(f"\n[Copy] Syncing dataset to local project folder '{dest_path.resolve()}'...")
        dest_path.mkdir(parents=True, exist_ok=True)

        for item in cached_path.iterdir():
            target = dest_path / item.name
            if item.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)

        print(f"[Done] Local dataset ready at: {dest_path.resolve()}")
        return dest_path

    return cached_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Dogs and Cats Classifier dataset from Kaggle")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Kaggle dataset handle")
    parser.add_argument("--output-dir", type=str, default="data", help="Local directory to store dataset")
    parser.add_argument("--token", type=str, default=DEFAULT_TOKEN, help="Kaggle API token (KGAT_...)")
    parser.add_argument("--cache-only", action="store_true", help="Keep in cache only without copying locally")

    args = parser.parse_args()

    download_and_setup_dataset(
        dataset_handle=args.dataset,
        output_dir=args.output_dir,
        token=args.token,
        copy_to_local=not args.cache_only
    )
