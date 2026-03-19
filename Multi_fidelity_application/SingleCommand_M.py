"""
Single-command script to reproduce pressure coefficient prediction of DLR-F4 wing-body configuration on multi-fidelity data
This will produce Figs. 7 and 8

Requirements:
- Full TabArena baseline setup (conda environment named `mftabpfn-full`)
  Multi-fidelity Gaussian Process baselines setup (conda environment named `mftabpfn-gpr`)
  Please follow the installation instructions in the root README.md.

- Datasets
  All datasets used in this work are publicly available at:
  https://zenodo.org/records/18502924

  Download `Datasets.zip`, unzip it, and place the extracted `Datasets/`
  folder directly in the root directory of the repository.
"""

import subprocess
import sys
import urllib.request
import zipfile
import time
import re
from pathlib import Path


def download_datasets(root_path: Path):
    """Automatically download and extract Datasets.zip if the Datasets/ folder is missing."""
    datasets_dir = root_path / "Datasets"
    # Check if Datasets/ already exists and is not empty
    if datasets_dir.exists() and any(datasets_dir.iterdir()):
        print("Datasets folder already exists in root directory. Skipping download.\n")
        return
    print("Datasets folder not found in root directory.")
    print("Downloading from Zenodo[](https://zenodo.org/records/18502924)...")
    zip_url = "https://zenodo.org/records/18502924/files/Datasets.zip?download=1"
    zip_path = root_path / "Datasets.zip"
    # Download
    try:
        urllib.request.urlretrieve(zip_url, zip_path)
        print("Download completed.")
    except Exception as e:
        print(f"Download failed: {e}")
        sys.exit(1)
    # Extract
    print("Extracting Datasets.zip ...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(root_path)
        print(f"Datasets successfully extracted to: {datasets_dir}")
    except Exception as e:
        print(f"Extraction failed: {e}")
        sys.exit(1)
    # Clean up zip file
    try:
        zip_path.unlink()
    except:
        pass  # ignore if deletion fails


def prepare_plot_script(script_name: str):
    script_path = Path(script_name)
    temp_path = Path(script_name + ".tmp")
    content = script_path.read_text(encoding="utf-8")

    backend_inject = (
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
    )

    if "import matplotlib" not in content:
        content = backend_inject + content
    else:
        content = re.sub(r'^(import .*?matplotlib.*?$)', backend_inject + r'\1', content, flags=re.MULTILINE)

    content = re.sub(
        r'^(.*?)(SAVE_DIR|SAVE_DIR1|SAVE_DIR2)\s*=\s*.*?$',
        r'\1\2 = Path(".")',
        content,
        flags=re.MULTILINE
    )
    if "from pathlib import Path" not in content and "import Path" not in content:
        content = "from pathlib import Path\n" + content
    temp_path.write_text(content, encoding="utf-8")
    return str(temp_path)

def run_script(script_name: str):
    script_path = Path(script_name)
    if not script_path.exists():
        print(f"Error: {script_path} not found!")
        sys.exit(1)

    if script_name == "CP_DLRF4_Position_MFGPR_M.py":
        env_name = "mftabpfn-gpr"
    else:
        env_name = "mftabpfn-full"

    if script_name in ["CP_DLRF4_Position_M_Plot.py"]:
        run_path = prepare_plot_script(script_name)
        is_temp = True
    else:
        run_path = script_name
        is_temp = False
    print(f"\n{'='*80}")
    print(f"Running {script_name} ...")
    print(f"{'='*80}")

    cmd = f"conda run -n {env_name} --live-stream python {run_path}"
    try:
        subprocess.run(cmd, cwd=".", shell=True, check=True)
        print(f"{script_name} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Failed to run {script_name} in {env_name} environment")
        print(f"Command was: {cmd}")
        print(f"Please check that both environments exist and conda is in your PATH.")
        sys.exit(1)
    finally:
        if is_temp and Path(run_path).exists():
            Path(run_path).unlink(missing_ok=True)


def main():
    # Determine repository root
    root_path = Path(__file__).resolve().parent.parent

    # === AUTOMATIC DATASET DOWNLOAD ===
    download_datasets(root_path)

    # === Run the reproduction scripts ===
    scripts_to_run = [
        "CP_DLRF4_Position_M.py",           # Performance based on MFTabPFN/TabPFN-H/TabPFN-M/AutoGluon-H/AutoGluon-M
        "CP_DLRF4_Position_FNO_M.py",       # Performance based on TLFNO
        "CP_DLRF4_Position_MAHNN_M.py",     # Performance based on MAHNN
        "CP_DLRF4_Position_MFGPR_M.py",     # Performance based on MFGPR/NMFGPR
        "CP_DLRF4_Position_M_Plot.py",      # Performance plots
    ]

    print("Starting reproduction of multi-fidelity prediction results...\n")

    for script in scripts_to_run:
        run_script(script)
        time.sleep(1)

    print("="*80)
    print("Multi-fidelity prediction problems finished!")
    print("="*80)
    print("\nYou can now view the figures directly in the repository")

if __name__ == "__main__":
    main()

