import argparse
import os

import pandas as pd
import torch

from src.oasis3_parser import BIDSFilenameParser, VolumeQualityFilter, OASIS3ScanIndexer
from src.mri_pipeline import (
    SitkVolumeReader, MNI152TemplateProvider, HDBETSkullStripper, N4BiasCorrector,
    MNIRegistrar, BrainZScoreNormalizer, ProcessedVolumeWriter, MRIPreprocessingPipeline,
)

# --- Configuration ---
MRI_DIR = 'data/mri_pronte'
LABELS_CSV = 'data/processed/oasis3_ad_risk.csv'
OUTPUT_DIR = 'data/preprocessed'
TEMPLATE_DIR = 'data/templates'
FAILURE_LOG = 'data/processed/preprocessing_failures.csv'


def ordered_scans(indexer: OASIS3ScanIndexer) -> pd.DataFrame:
    '''
    All indexed sessions, with the currently labelled ones first so they are ready for training sooner.
    '''
    scans = indexer.index()[['ID', 'Path']]
    if os.path.exists(LABELS_CSV):
        labelled_ids = set(pd.read_csv(LABELS_CSV)['ID'])
        scans = scans.assign(Priority=~scans['ID'].isin(labelled_ids)).sort_values(['Priority', 'ID'])
    return scans.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description='Batch preprocessing of OASIS-3 T1w volumes.')
    parser.add_argument('--limit', type=int, default=None, help='Process at most N scans (for quick tests).')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    scans = ordered_scans(OASIS3ScanIndexer(MRI_DIR, BIDSFilenameParser(), VolumeQualityFilter()))
    if args.limit is not None:
        scans = scans.head(args.limit)

    template = MNI152TemplateProvider(cache_dir=TEMPLATE_DIR).load()
    pipeline = MRIPreprocessingPipeline(
        reader=SitkVolumeReader(),
        steps=[
            HDBETSkullStripper(device=device),
            N4BiasCorrector(),
            MNIRegistrar(template=template),
            BrainZScoreNormalizer(),
        ],
        writer=ProcessedVolumeWriter(output_dir=OUTPUT_DIR),
        failure_log_path=FAILURE_LOG,
    )
    pipeline.run(scans)


if __name__ == '__main__':
    main()
