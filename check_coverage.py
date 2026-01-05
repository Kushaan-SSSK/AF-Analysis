import pandas as pd
import os
import glob

annot_path = r"GroundtruthverifiedEKGData\EKG_Annotations.xlsx"
df = pd.read_excel(annot_path)
print("Columns:", df.columns.tolist())
# Assume Col 0 is filename
keys = df.iloc[:,0].astype(str).tolist()

# Normalize keys
norm_keys = []
for k in keys:
    base = os.path.basename(k).replace('.iwxdata', '').replace('_Export', '').strip().lower()
    norm_keys.append(base)

print(f"Loaded {len(norm_keys)} keys.")

# Check EKG data
ekg_files = glob.glob(r"EKG data\*_Export.xls")
print(f"Found {len(ekg_files)} files in EKG data.")

matches = 0
for f in ekg_files:
    base = os.path.basename(f).replace('.iwxdata', '').replace('_Export', '').replace('.xls', '').strip().lower()
    # Simple check
    found = False
    for k in norm_keys:
        if base in k or k in base:
            found = True
            break
    if found:
        matches += 1
    else:
        print(f"No match for: {base}")

print(f"Matches found: {matches}/{len(ekg_files)}")
