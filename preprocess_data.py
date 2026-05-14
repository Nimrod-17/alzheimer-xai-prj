from src.data_loader import ExcelClinicalParser, MRIFileResolver, MRIDataLoader
from src.skull_stripping import SkullStripper, CleanedMRIWriter, SkullStrippingPipeline

# --- Configuration ---
EXCEL_PATH = 'data/Demographic and Clinical Data/oasis_cross-sectional.xlsx'
DATA_DIR = 'data'
CLEANED_DIR = 'data/cleaned_data'


def main():
    # 1. Load the dataframe with matched MRI paths
    parser = ExcelClinicalParser(excel_path=EXCEL_PATH)
    resolver = MRIFileResolver(
        base_data_dir=DATA_DIR,
        search_pattern="disc*/{id}/PROCESSED/MPRAGE/T88_111/*.hdr"
    )
    data_loader = MRIDataLoader(parser=parser, resolver=resolver)
    df = data_loader.load_data()

    # 2. Build and run the skull stripping pipeline (all dependencies injected)
    stripper = SkullStripper(use_gpu=True, use_tta=False, verbose=True)
    writer = CleanedMRIWriter(output_dir=CLEANED_DIR)
    pipeline = SkullStrippingPipeline(stripper=stripper, writer=writer)

    pipeline.run(df)


if __name__ == '__main__':
    main()