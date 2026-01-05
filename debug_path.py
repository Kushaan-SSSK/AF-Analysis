import os
path = r"c:\Users\kusha\Downloads\AF Analysis\GroundtruthverifiedEKGData\EKG_Annotations.xlsx"
print(f"Path: {path}")
print(f"Exists: {os.path.exists(path)}")

base_dir = r"c:\Users\kusha\Downloads\AF Analysis"
gt_dir = os.path.join(base_dir, "GroundtruthverifiedEKGData")
print(f"GT Dir Exists: {os.path.exists(gt_dir)}")
print(f"Contents: {os.listdir(gt_dir) if os.path.exists(gt_dir) else 'N/A'}")
