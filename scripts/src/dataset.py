import re
import shutil
from pathlib import Path
import imagehash
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

# --- Configuration Paths ---
source_root = Path("../../../chest_xray")
merged_root = Path("../../../chest_xray_merged")
clean_root = Path("../../../chest_xray_clean")

classes = ["normal", "pneumonia"]

# ==========================================
# STEP 1: Merge train, test, and val splits
# ==========================================
print("Step 1: Merging train, test, and val folders...")
if merged_root.exists():
  shutil.rmtree(merged_root)

for class_name in classes:
  target_class_dir = merged_root / class_name
  target_class_dir.mkdir(parents=True, exist_ok=True)

  for split in ["train", "test", "val"]:
    split_class_dir = source_root / split / class_name

    if not split_class_dir.exists():
      continue

    for img_path in split_class_dir.iterdir():
      if img_path.is_file():
        # Prefix with split name to avoid filename collisions
        dest_filename = f"{split}_{img_path.name}"
        shutil.copy2(img_path, target_class_dir / dest_filename)

print(f"-> Merged dataset created at: {merged_root.resolve()}")


# ==========================================
# STEP 2 & 4: Deduplicate Pneumonia by Name
# ==========================================
def process_pneumonia():
  print("\nStep 2 & 4: Deduplicating PNEUMONIA by patient ID...")
  source_dir = merged_root / "pneumonia"
  dest_dir = clean_root / "pneumonia"
  dest_dir.mkdir(parents=True, exist_ok=True)

  seen_patients = set()
  kept_count = 0
  skipped_count = 0

  for img_path in source_dir.glob("*"):
    if not img_path.is_file():
      continue

    filename = img_path.name
    # Extract explicit patient ID (e.g., person1945 from train_person1945_...)
    match = re.search(r"(person\d+)", filename)

    if match:
      patient_id = match.group(1)
      if patient_id not in seen_patients:
        seen_patients.add(patient_id)
        shutil.copy2(img_path, dest_dir / filename)
        kept_count += 1
      else:
        skipped_count += 1
    else:
      # Fallback if pattern doesn't match
      shutil.copy2(img_path, dest_dir / filename)
      kept_count += 1

  print(
      f"-> [PNEUMONIA] Kept {kept_count} unique patients (removed"
      f" {skipped_count} multi-scan duplicates)."
  )


# ==========================================
# STEP 3 & 4: Deduplicate Normal via dHash
# ==========================================
def process_normal(threshold=0):
  print(
      "\nStep 3 & 4: Deduplicating NORMAL via perceptual hashing (Threshold ="
      f" {threshold})..."
  )
import re
import shutil
from pathlib import Path
import imagehash
from PIL import Image

# Configuration paths
source_root = Path("../../../chest_xray_merged")
clean_root = Path("../../../chest_xray_clean")

# Initialize clean output directories
for class_name in ["NORMAL", "PNEUMONIA"]:
  (clean_root / class_name).mkdir(parents=True, exist_ok=True)


def process_pneumonia():
  source_dir = source_root / "PNEUMONIA"
  dest_dir = clean_root / "PNEUMONIA"

  seen_patients = set()
  kept_count = 0
  skipped_count = 0

  for img_path in source_dir.glob("*"):
    if not img_path.is_file():
      continue

    filename = img_path.name
    # Extract explicit patient ID (e.g., person1945 from person1945_bacteria_4872.jpeg)
    match = re.match(r"(person\d+)", filename)

    if match:
      patient_id = match.group(1)
      if patient_id not in seen_patients:
        seen_patients.add(patient_id)
        shutil.copy2(img_path, dest_dir / filename)
        kept_count += 1
      else:
        # Skip extra scans for the same patient
        skipped_count += 1
    else:
      # Fallback if filename pattern differs unexpectedly
      shutil.copy2(img_path, dest_dir / filename)
      kept_count += 1

  print(
      f"🛡️ [PNEUMONIA] Kept {kept_count} unique patients (removed"
      f" {skipped_count} multi-scan duplicates)."
  )

  source_dir = merged_root / "normal"
  dest_dir = clean_root / "normal"
  dest_dir.mkdir(parents=True, exist_ok=True)

  hashes = {}
  kept_count = 0
  duplicate_count = 0

  for img_path in source_dir.glob("*"):
    if not img_path.is_file():
      continue

    try:
      pil_img = Image.open(img_path).convert("L")
      h = imagehash.dhash(pil_img)

      is_duplicate = False
      for existing_hash in hashes:
        if h - existing_hash <= threshold:
          is_duplicate = True
          break

      if not is_duplicate:
        hashes[h] = img_path
        shutil.copy2(img_path, dest_dir / img_path.name)
        kept_count += 1
      else:
        duplicate_count += 1

    except Exception as e:
      print(f"Skipping {img_path.name} due to error: {e}")
      continue

  print(
      f"-> [NORMAL] Kept {kept_count} unique images (removed"
      f" {duplicate_count} structural duplicates)."
  )


# ==========================================
# Execution & Step 5: Final Clean Dataset Ready
# ==========================================
if __name__ == "__main__":
  if clean_root.exists():
    shutil.rmtree(clean_root)

  process_pneumonia()
  process_normal(threshold=0)  # threshold=0 ensures conservative, exact hash matching

  print(
      "\nStep 5: Pipeline complete! Clean, non-split dataset ready at:"
      f" {clean_root.resolve()}"
  )
