import pandas as pd
import os

file_path = r"c:\Users\kusha\Downloads\AF Analysis\GroundtruthverifiedEKGData\EKG_Annotations.xlsx"

try:
    df = pd.read_excel(file_path)
    print("Columns:", df.columns.tolist())
    
    # Rename for easier access
    df.columns = ['Filename', 'AF', 'RR_irregular', 'P_wave_loss', 'Other']
    
    print("\nUnique values in 'AF' column:")
    print(df['AF'].unique())
    
    print("\nFirst 10 Filenames:")
    for f in df['Filename'].head(10):
        print(f)
        
except Exception as e:
    print(f"Error reading excel: {e}")
