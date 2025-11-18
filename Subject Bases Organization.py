import os
import shutil

# --- Config ---
DATASET_PATH = r"C:/Users/kapla/Desktop/ZEKE NUEMove/MachineLearningDataSet/Walking Dataset"
DEST_ROOT    = r"C:/Users/kapla/Desktop/ZEKE NUEMove/MachineLearningDataSet/Subject Based Walking 2"
CHUNK_SIZE   = 8
# --------------

def split_list(lst, chunk):
    return [lst[i:i+chunk] for i in range(0, len(lst), chunk)]

# Get only .mat files (ignore folders), sorted for reproducibility
all_entries = os.listdir(DATASET_PATH)
mat_files = sorted(
    [f for f in all_entries if f.lower().endswith(".mat") and os.path.isfile(os.path.join(DATASET_PATH, f))]
)

print(f"Found {len(mat_files)} .mat files")
subjects = split_list(mat_files, CHUNK_SIZE)

os.makedirs(DEST_ROOT, exist_ok=True)

for idx, files_chunk in enumerate(subjects, 1):  # Subject1, Subject2, ...
    subject_dir = os.path.join(DEST_ROOT, f"Subject{idx}")
    os.makedirs(subject_dir, exist_ok=True)

    for fname in files_chunk:
        src = os.path.join(DATASET_PATH, fname)
        # Copy into the subject folder (keep original filename)
        shutil.copy2(src, subject_dir)

    print(f"Subject {idx}: copied {len(files_chunk)} files "
          f"({idx}/{len(subjects)} = {idx/len(subjects):.2%})")

print("Done!")