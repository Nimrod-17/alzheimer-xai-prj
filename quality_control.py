import nibabel as nib
import numpy as np

from src.mri_pipeline import ProcessedVolumeWriter
from src.preprocessing import CropResizeLoader
from src.quality_control import (
    TemplateSimilarityMetric, BrainVolumeMetric, AffineScaleMetric, CropCoverageMetric,
    QCOutlierDetector, QCSnapshotRenderer, QualityControlRunner,
)

# --- Configuration ---
PREPROCESSED_DIR = 'data/preprocessed'
TEMPLATE_IMAGE = 'data/templates/mni152_brain_1mm.nii.gz'
TEMPLATE_MASK = 'data/templates/mni152_mask_1mm.nii.gz'
REPORT_PATH = 'data/processed/qc_report.csv'
SNAPSHOT_DIR = 'data/processed/qc_snapshots'

# Clearly broken outputs, whatever the rest of the dataset looks like.
# Calibrated on visually checked cases: good registrations reach Dice ~0.95 and NCC ~0.4-0.55,
# failed ones (shifted, rotated or mis-scaled brains) fell to Dice < 0.90 and NCC < 0.2.
HARD_LIMITS = {
    'template_dice': (0.90, 1.0),
    'template_ncc': (0.25, 1.0),
    'brain_volume_cm3': (800, 2000),
    'scale_min': (0.6, 1.5),
    'scale_max': (0.6, 1.5),
    'crop_loss': (0.0, 0.02),
}
# Subtler outliers, relative to the dataset distribution
ROBUST_METRICS = ('template_dice', 'template_ncc', 'brain_volume_cm3', 'scale_anisotropy')


def main():
    template_image = np.asarray(nib.load(TEMPLATE_IMAGE).get_fdata(dtype=np.float32))
    template_mask = np.asarray(nib.load(TEMPLATE_MASK).get_fdata()) > 0

    runner = QualityControlRunner(
        volume_store=ProcessedVolumeWriter(PREPROCESSED_DIR),
        metrics=[
            TemplateSimilarityMetric(template_image, template_mask),
            BrainVolumeMetric(),
            AffineScaleMetric(),
            CropCoverageMetric(CropResizeLoader().crop),
        ],
        detector=QCOutlierDetector(HARD_LIMITS, ROBUST_METRICS),
        renderer=QCSnapshotRenderer(template_mask, SNAPSHOT_DIR),
        report_path=REPORT_PATH,
    )
    report = runner.run()

    print("\nMetric distribution:")
    print(report.drop(columns=['ID', 'qc_flagged', 'qc_reasons']).describe(percentiles=[0.01, 0.5, 0.99]).round(3).to_string())
    flagged = report[report['qc_flagged']]
    if len(flagged):
        print("\nFlagged volumes:")
        print(flagged[['ID', 'qc_reasons']].to_string(index=False))


if __name__ == '__main__':
    main()
