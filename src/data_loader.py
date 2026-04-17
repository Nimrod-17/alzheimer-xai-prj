import os
import glob
import pandas as pd

def create_mri_dataset(excel_path, base_data_dir):
    """
    Loads clinical data, cleans labels, and matches them with MRI file paths
    """
    
    print("Loading Excel file...")
    df = pd.read_excel(excel_path)
    print(f"Excel loaded seccessfully. Found {len(df)} entries.")

    df = df.dropna(subset=['CDR']).copy()
    df['CDR'] = df['CDR'].astype(str).str.replace(',', '.').astype(float)
    df['Label'] = (df['CDR'] > 0.0).astype(int)

    print("Starting 3D image search...")
    valid_data = []

    for index, row in df.iterrows():
        patient_id = row['ID']
        search_pattern = os.path.join(base_data_dir, "disc*", patient_id, "PROCESSED", "MPRAGE", "T88_111", "*.hdr")

        hdr_files = glob.glob(search_pattern)

        if len(hdr_files) > 0:

            valid_data.append({
                'ID': patient_id,
                'Path': hdr_files[0],
                'Label': row['Label']
            })

        if index > 0 and index % 50 == 0:
            print(f"Processed {index} patients...")

    print(f"Finished searching. Found {len(valid_data)} valid entries with matching MRI files.")

    return pd.DataFrame(valid_data)