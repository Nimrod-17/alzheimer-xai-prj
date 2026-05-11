import os
import glob
import pandas as pd
from src.interfaces import DataLoaderInterface

class ExcelClinicalParser:
    '''
    Read and clean clinical data from an excel file
    '''
    def __init__(self, excel_path: str):
        self.excel_path = excel_path

    def parse(self) -> pd.DataFrame:
        print("Loading Excel file...")
        df = pd.read_excel(self.excel_path)
        print(f"Excel loaded seccessfully. Found {len(df)} entries.")

        df = df.dropna(subset=['CDR']).copy()
        df['CDR'] = df['CDR'].astype(str).str.replace(',', '.').astype(float)
        df['Label'] = (df['CDR'] > 0.0).astype(int)

        return df

class MRIFileResolver:
    '''
    Locate MRI files on disk given a patient ID.
    The search pattern is injectable, allowing extension without modification.
    '''
    def __init__(self, base_data_dir: str, search_pattern: str = "disc*/{id}/PROCESSED/MPRAGE/T88_111/*.hdr"):
        self.base_data_dir = base_data_dir
        self.search_pattern = search_pattern

    def resolve(self, patient_id: str) -> str | None:
        pattern = os.path.join(
            self.base_data_dir,
            self.search_pattern.format(id=patient_id)
        )
        files = glob.glob(pattern)
        return files[0] if files else None
    
class MRIDataLoader(DataLoaderInterface):
    '''
    Depends on abstractions (injected parser and resolver), not on concrete implementations.
    '''
    def __init__(self, parser: ExcelClinicalParser, resolver: MRIFileResolver):
        self.parser = parser
        self.resolver = resolver

    def load_data(self) -> pd.DataFrame:
        df = self.parser.parse()

        print("Starting 3D image search...")
        print(f"Base directory: {self.resolver.base_data_dir}") 

        valid_data = []

        for index, row in df.iterrows():
            patient_id = row['ID']
            path = self.resolver.resolve(patient_id)

            if index < 3:  # stampa i primi 3 pattern per debug
                import os, glob
                pattern = os.path.join(
                    self.resolver.base_data_dir,
                    self.resolver.search_pattern.format(id=patient_id)
                )
                print(f"  Patient: {patient_id}")
                print(f"  Pattern: {pattern}")
                print(f"  Found: {glob.glob(pattern)}")

            if path:
                valid_data.append({
                    'ID': patient_id,
                    'Path': path,
                    'Label': row['Label']
                })

            if index > 0 and index % 50 == 0:
                print(f"Processed {index} patients...")

        print(f"Finished. Found {len(valid_data)} valid entries with matching MRI files.")
        return pd.DataFrame(valid_data)
