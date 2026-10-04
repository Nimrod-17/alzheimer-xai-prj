import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk

from src.interfaces import QCMetricInterface
from src.mri_pipeline import ProcessedVolumeWriter


class TemplateSimilarityMetric(QCMetricInterface):
    '''
    Registration quality: correlation of intensities with the MNI template
    and Dice overlap between the subject brain mask and the template brain mask.
    '''
    def __init__(self, template_image: np.ndarray, template_mask: np.ndarray):
        self.template_image = template_image
        self.template_mask = template_mask

    def compute(self, image: np.ndarray, mask: np.ndarray, affine_matrix: np.ndarray) -> dict[str, float]:
        overlap = self.template_mask & mask
        ncc = np.corrcoef(image[overlap], self.template_image[overlap])[0, 1] if overlap.any() else 0.0
        dice = 2 * overlap.sum() / (self.template_mask.sum() + mask.sum())
        return {'template_ncc': float(ncc), 'template_dice': float(dice)}


class BrainVolumeMetric(QCMetricInterface):
    '''
    Skull-stripping plausibility: brain volume in native space (cm³), recovered from the
    MNI-space mask and the determinant of the registration.
    Too large suggests leftover skull/dura, too small suggests over-stripping.
    '''
    def compute(self, image: np.ndarray, mask: np.ndarray, affine_matrix: np.ndarray) -> dict[str, float]:
        native_mm3 = mask.sum() * abs(np.linalg.det(affine_matrix))  # MNI grid is 1 mm isotropic
        return {'brain_volume_cm3': float(native_mm3 / 1000)}


class AffineScaleMetric(QCMetricInterface):
    '''
    Registration plausibility: scaling factors of the affine (singular values).
    Extreme or strongly anisotropic scaling reveals a failed registration.
    '''
    def compute(self, image: np.ndarray, mask: np.ndarray, affine_matrix: np.ndarray) -> dict[str, float]:
        scales = np.linalg.svd(affine_matrix, compute_uv=False)
        return {'scale_min': float(scales.min()), 'scale_max': float(scales.max()),
                'scale_anisotropy': float(scales.max() / scales.min())}


class CropCoverageMetric(QCMetricInterface):
    '''
    Fraction of brain voxels that fall outside the crop box used to build the network input.
    '''
    def __init__(self, crop: tuple[slice, ...]):
        self.crop = crop

    def compute(self, image: np.ndarray, mask: np.ndarray, affine_matrix: np.ndarray) -> dict[str, float]:
        total = mask.sum()
        return {'crop_loss': float(1 - mask[self.crop].sum() / total) if total else 1.0}


class QCOutlierDetector:
    '''
    Flag volumes using hard limits (clearly broken outputs) and robust z-scores (median/MAD),
    which adapt to the dataset and catch subtler outliers without assuming a distribution.
    '''
    def __init__(self, hard_limits: dict[str, tuple[float, float]], robust_metrics: tuple[str, ...],
                 robust_z_threshold: float = 4.0):
        self.hard_limits = hard_limits
        self.robust_metrics = robust_metrics
        self.robust_z_threshold = robust_z_threshold

    def flag(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        reasons = pd.Series([[] for _ in range(len(df))], index=df.index)

        for metric, (low, high) in self.hard_limits.items():
            outside = (df[metric] < low) | (df[metric] > high)
            for idx in df.index[outside]:
                reasons[idx].append(f'{metric}={df.at[idx, metric]:.3g} outside [{low}, {high}]')

        for metric in self.robust_metrics:
            median = df[metric].median()
            mad = (df[metric] - median).abs().median() * 1.4826  # consistent with std for normal data
            if mad == 0:
                continue
            z = (df[metric] - median) / mad
            for idx in df.index[z.abs() > self.robust_z_threshold]:
                reasons[idx].append(f'{metric}={df.at[idx, metric]:.3g} (robust z {z[idx]:+.1f})')

        df['qc_flagged'] = reasons.map(bool)
        df['qc_reasons'] = reasons.map('; '.join)
        return df


class QCSnapshotRenderer:
    '''
    Save orthogonal mid-slices of a volume with the template brain outline in red, for visual review.
    '''
    def __init__(self, template_mask: np.ndarray, output_dir: str):
        self.template_mask = template_mask
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def render(self, scan_id: str, image: np.ndarray, title: str) -> Path:
        centre = [size // 2 for size in image.shape]
        fig, axes = plt.subplots(1, 3, figsize=(11, 4))

        for axis, (volume_slice, outline_slice) in zip(axes, self._slices(image, centre)):
            axis.imshow(np.rot90(volume_slice), cmap='gray')
            axis.contour(np.rot90(outline_slice), levels=[0.5], colors='r', linewidths=0.6)
            axis.axis('off')

        fig.suptitle(f'{scan_id}\n{title}', fontsize=8)
        fig.tight_layout()
        output_path = self.output_dir / f'{scan_id}.png'
        fig.savefig(output_path, dpi=70)
        plt.close(fig)
        return output_path

    def _slices(self, image: np.ndarray, centre: list[int]):
        x, y, z = centre
        template = self.template_mask.astype(float)
        return [(image[x], template[x]), (image[:, y], template[:, y]), (image[:, :, z], template[:, :, z])]


class QualityControlRunner:
    '''
    Compute QC metrics for every preprocessed volume, flag outliers and render snapshots of flagged ones.
    Incremental: metrics already stored in the report are reused, only new volumes are measured.
    '''
    def __init__(self, volume_store: ProcessedVolumeWriter, metrics: list[QCMetricInterface],
                 detector: QCOutlierDetector, renderer: QCSnapshotRenderer, report_path: str):
        self.volume_store = volume_store
        self.metrics = metrics
        self.detector = detector
        self.renderer = renderer
        self.report_path = report_path

    def run(self) -> pd.DataFrame:
        scan_ids = sorted(path.name.replace('_T1w_mni.nii.gz', '')
                          for path in self.volume_store.output_dir.glob('*_T1w_mni.nii.gz'))

        previous = self._load_previous()
        new_ids = [scan_id for scan_id in scan_ids if scan_id not in previous.index]
        print(f"--- Quality control: {len(previous)} cached, {len(new_ids)} to measure ---")

        rows = []
        for count, scan_id in enumerate(new_ids, start=1):
            rows.append({'ID': scan_id, **self._measure(scan_id)})
            if count % 50 == 0:
                print(f"  measured {count}/{len(new_ids)}", flush=True)

        measured = pd.concat([previous.reset_index(), pd.DataFrame(rows)], ignore_index=True)
        measured = measured[measured['ID'].isin(scan_ids)]  # forget volumes deleted from disk
        report = self.detector.flag(measured.drop(columns=['qc_flagged', 'qc_reasons'], errors='ignore'))

        os.makedirs(os.path.dirname(self.report_path), exist_ok=True)
        report.sort_values(['qc_flagged', 'ID'], ascending=[False, True]).to_csv(self.report_path, index=False)

        flagged = report[report['qc_flagged']]
        for row in flagged.itertuples():
            image = self._load(self.volume_store.image_path(row.ID))
            self.renderer.render(row.ID, image, row.qc_reasons)

        print(f"QC complete: {len(flagged)}/{len(report)} volumes flagged. Report → {self.report_path}")
        return report

    def _measure(self, scan_id: str) -> dict[str, float]:
        image = self._load(self.volume_store.image_path(scan_id))
        mask = self._load(self.volume_store.mask_path(scan_id)) > 0
        transform = sitk.ReadTransform(str(self.volume_store.transform_path(scan_id)))
        affine_matrix = np.array(sitk.AffineTransform(transform).GetMatrix()).reshape(3, 3)

        results = {}
        for metric in self.metrics:
            results.update(metric.compute(image, mask, affine_matrix))
        return results

    def _load_previous(self) -> pd.DataFrame:
        if not os.path.exists(self.report_path):
            return pd.DataFrame().rename_axis('ID')
        return pd.read_csv(self.report_path).set_index('ID')

    @staticmethod
    def _load(path: Path) -> np.ndarray:
        return np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32))
