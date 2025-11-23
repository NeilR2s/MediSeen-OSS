# prepare_gt.py
import csv
import os
import random

# --- Configuration ---
gt_file_path = 'final_dataset/labels.txt'
output_dir = 'output'
train_csv_filename = 'train.csv'
val_csv_filename = 'val.csv'
split_ratio = 0.8  # 80% for training, 20% for validation

print("🚀 Starting data preparation...")

if not os.path.exists(gt_file_path):
    print(f"❌ Error: Ground truth file not found at '{gt_file_path}'")
    exit()

# Read the tab-separated gt.txt file
lines = []
try:
    with open(gt_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            parts = line.strip().split(None, 1)
            if len(parts) == 2 and parts[0] and parts[1]:
                lines.append(parts)
            else:
                print(f"⚠️ Warning: Skipping malformed or empty line #{i+1}: {line.strip()}")
except Exception as e:
    print(f"❌ An error occurred while reading {gt_file_path}: {e}")
    exit()

# Shuffle and split the data
random.shuffle(lines)
split_index = int(len(lines) * split_ratio)
train_data = lines[:split_index]
val_data = lines[split_index:]

print(f"📊 Total valid samples: {len(lines)}")
print(f"   - Training samples: {len(train_data)}")
print(f"   - Validation samples: {len(val_data)}")

# Function to write data to a CSV file
def write_csv(file_path, data):
    try:
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['filename', 'label'])  # Write header
            writer.writerows(data)
        print(f"✅ Successfully created '{file_path}'")
    except Exception as e:
        print(f"❌ Failed to write to {file_path}: {e}")

# Write the training and validation CSV files
os.makedirs(output_dir, exist_ok=True)
write_csv(os.path.join(output_dir, train_csv_filename), train_data)
write_csv(os.path.join(output_dir, val_csv_filename), val_data)

print("\nData preparation complete. Run the following commands to create the LMDB datasets.")
print("\npython3 create_lmdb_dataset.py create \
  --imagePathOnly final_dataset/ \
  --gtFile output/train.csv \
  --outputPath data/train")
print("\npython3 create_lmdb_dataset.py create \
  --imagePathOnly final_dataset/ \
  --gtFile output/val.csv \
  --outputPath data/val")