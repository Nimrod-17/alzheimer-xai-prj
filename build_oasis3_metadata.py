import os

from src.oasis3_parser import (
    BIDSFilenameParser, VolumeQualityFilter, OASIS3ScanIndexer, OASIS3ClinicalParser,
    NearestVisitMatcher, ADDiagnosisRule, ADRiskLabeler, AllScansSelector, OASIS3DataLoader,
)

# --- Configuration ---
MRI_DIR = 'data/mri_pronte'
CDR_CSV_PATH = (
    r'C:\Users\collu\Desktop\OASIS3_data_files\scans'
    r'\UDSb4-Form_B4__Global_Staging__CDR__Standard_and_Supplemental'
    r'\resources\csv\files\OASIS3_UDSb4_cdr.csv'
)
OUTPUT_PATH = 'data/processed/oasis3_ad_risk.csv'
MAX_GAP_DAYS = 180              # MRI-to-clinical-visit tolerance
CONVERSION_HORIZON_DAYS = 1825  # converters to AD within 5 years count as positive
MIN_FOLLOWUP_DAYS = 1095        # normal subjects need 3 years of follow-up to count as stable


def main():
    # All scans are kept: subject-level splitting and sampling are handled at training time
    loader = OASIS3DataLoader(
        indexer=OASIS3ScanIndexer(MRI_DIR, BIDSFilenameParser(), VolumeQualityFilter()),
        clinical_parser=OASIS3ClinicalParser(CDR_CSV_PATH),
        matcher=NearestVisitMatcher(max_gap_days=MAX_GAP_DAYS),
        labeler=ADRiskLabeler(
            ad_rule=ADDiagnosisRule(),
            conversion_horizon_days=CONVERSION_HORIZON_DAYS,
            min_followup_days=MIN_FOLLOWUP_DAYS,
        ),
        selector=AllScansSelector(),
    )
    df = loader.load_data()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved → {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
