import os
import json
from tqdm import tqdm  

root_dir = r"training_results\smc_trace_sweeps"

best_ll = float("-inf")   # find MOST POSITIVE
best_path = None


all_files = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    if "output_loglik.json" in filenames:
        all_files.append(os.path.join(dirpath, "output_loglik.json"))

# Now iterate with progress bar
for file_path in tqdm(all_files, desc="Scanning loglik files"):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        total_ll = data.get("total_ll", None)

        if total_ll is None:
            continue

        # maximize log-likelihood
        if total_ll > best_ll:
            best_ll = total_ll
            best_path = file_path

    except Exception as e:
        print(f"Error reading {file_path}: {e}")

print("=" * 60)
print("BEST (MOST POSITIVE) TOTAL LOG-LIKELIHOOD:")
print(best_ll)

print("\nFILE PATH:")
print(best_path)

print("\nHYPERPARAM SET:")
if best_path:
    print(os.path.basename(os.path.dirname(best_path)))
print("=" * 60)