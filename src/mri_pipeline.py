import csv
import os
import time
from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

from src.brain_volume import BrainVolume, sitk_to_nifti
from src.interfaces import VolumeStepInterface


class SitkVolumeReader:
    '''
    Load a NIfTI file as a float32 SimpleITK image wrapped in a BrainVolume.
    '''
    def read(self, file_path: str) -> BrainVolume:
        image = sitk.ReadImage(file_path, sitk.sitkFloat32)
        if image.GetDimension() == 4:  # singleton time axis
            image = image[:, :, :, 0]
        return BrainVolume(image=image)


class MNI152TemplateProvider:
    '''
    Provide the skull-stripped ICBM 2009 MNI152 template bundled with nilearn.
    The template is materialized once on disk and reused afterwards.
    '''
    def __init__(self, cache_dir: str = 'data/templates', resolution_mm: int = 1):
        self.cache_dir = Path(cache_dir)
        self.resolution_mm = resolution_mm

    def load(self) -> BrainVolume:
        image_path = self.cache_dir / f'mni152_brain_{self.resolution_mm}mm.nii.gz'
        mask_path = self.cache_dir / f'mni152_mask_{self.resolution_mm}mm.nii.gz'

        if not (image_path.exists() and mask_path.exists()):
            self._materialize(image_path, mask_path)

        return BrainVolume(
            image=sitk.ReadImage(str(image_path), sitk.sitkFloat32),
            mask=sitk.ReadImage(str(mask_path), sitk.sitkUInt8),
        )

    def _materialize(self, image_path: Path, mask_path: Path) -> None:
        from nilearn import datasets

        template = datasets.load_mni152_template(resolution=self.resolution_mm)
        mask = datasets.load_mni152_brain_mask(resolution=self.resolution_mm)
        mask_data = (mask.get_fdata() > 0).astype(np.uint8)
        brain_data = (template.get_fdata() * mask_data).astype(np.float32)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(brain_data, template.affine), str(image_path))
        nib.save(nib.Nifti1Image(mask_data, template.affine), str(mask_path))


class HDBETSkullStripper(VolumeStepInterface):
    '''
    Brain extraction with HD-BET 2.x, run in memory through its nnU-Net predictor
    (no temporary files or worker processes per volume).
    '''
    def __init__(self, device: torch.device, use_tta: bool = False):
        from HD_BET.hd_bet_prediction import get_hdbet_predictor
        self.predictor = get_hdbet_predictor(use_tta=use_tta, device=device, verbose=False)
        self.predictor.allow_tqdm = False

    def apply(self, volume: BrainVolume) -> BrainVolume:
        # Same layout produced by nnU-Net's SimpleITKIO: (channel, z, y, x) and spacing in (z, y, x)
        data = sitk.GetArrayFromImage(volume.image)[np.newaxis].astype(np.float32)
        properties = {'spacing': list(volume.image.GetSpacing())[::-1]}

        segmentation = self.predictor.predict_single_npy_array(data, properties)

        mask = sitk.GetImageFromArray(segmentation.astype(np.uint8))
        mask.CopyInformation(volume.image)
        return replace(volume, image=sitk.Mask(volume.image, mask), mask=mask)


class N4BiasCorrector(VolumeStepInterface):
    '''
    N4 bias field correction restricted to the brain mask.
    The field is estimated on a downsampled copy and applied at full resolution.
    '''
    def __init__(self, shrink_factor: int = 4, iterations: tuple[int, ...] = (50, 50, 50, 50)):
        self.shrink_factor = shrink_factor
        self.iterations = list(iterations)

    def apply(self, volume: BrainVolume) -> BrainVolume:
        shrink = [self.shrink_factor] * 3
        small_image = sitk.Shrink(volume.image, shrink)
        small_mask = sitk.Shrink(volume.mask, shrink)

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations(self.iterations)
        corrector.Execute(small_image, small_mask)

        log_bias = sitk.Cast(corrector.GetLogBiasFieldAsImage(volume.image), sitk.sitkFloat32)
        corrected = sitk.Cast(sitk.Mask(volume.image / sitk.Exp(log_bias), volume.mask), sitk.sitkFloat32)
        return replace(volume, image=corrected)


class MNIRegistrar(VolumeStepInterface):
    '''
    Linear registration to the MNI152 template: a rigid stage followed by a full affine stage,
    both driven by Mattes mutual information on skull-stripped images.
    RegularStepGradientDescent is used because plain gradient descent got stuck in wrong
    alignments on a fraction of scans (found by quality control).
    Image and mask are resampled onto the template grid; the transform is kept for later use.
    '''
    def __init__(self, template: BrainVolume, sampling_percentage: float = 0.25, seed: int = 42):
        self.template = template
        self.sampling_percentage = sampling_percentage
        self.seed = seed

    def apply(self, volume: BrainVolume) -> BrainVolume:
        fixed, moving = self.template.image, sitk.Cast(volume.image, sitk.sitkFloat32)

        rigid = sitk.Euler3DTransform(sitk.CenteredTransformInitializer(
            fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS
        ))
        self._register(fixed, moving, rigid)

        affine = sitk.AffineTransform(3)
        affine.SetCenter(rigid.GetCenter())
        affine.SetMatrix(rigid.GetMatrix())
        affine.SetTranslation(rigid.GetTranslation())
        self._register(fixed, moving, affine)

        image = sitk.Resample(moving, fixed, affine, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        mask = sitk.Resample(volume.mask, fixed, affine, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        return BrainVolume(image=image, mask=mask, transform=affine)

    def _register(self, fixed: sitk.Image, moving: sitk.Image, transform: sitk.Transform) -> None:
        '''Optimizes the given transform in place.'''
        registration = sitk.ImageRegistrationMethod()
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(self.sampling_percentage, self.seed)
        registration.SetInterpolator(sitk.sitkLinear)
        registration.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0, minStep=1e-4, numberOfIterations=300, relaxationFactor=0.5,
        )
        registration.SetOptimizerScalesFromPhysicalShift()
        registration.SetShrinkFactorsPerLevel([4, 2, 1])
        registration.SetSmoothingSigmasPerLevel([2, 1, 0])
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        registration.SetInitialTransform(transform, inPlace=True)
        registration.Execute(fixed, moving)


class BrainZScoreNormalizer(VolumeStepInterface):
    '''
    Z-score intensities using brain voxels only; the background is set to 0.
    '''
    def apply(self, volume: BrainVolume) -> BrainVolume:
        array = sitk.GetArrayFromImage(volume.image)
        brain = sitk.GetArrayFromImage(volume.mask) > 0

        values = array[brain]
        normalized = np.zeros_like(array, dtype=np.float32)
        normalized[brain] = (values - values.mean()) / max(values.std(), 1e-8)

        image = sitk.GetImageFromArray(normalized)
        image.CopyInformation(volume.image)
        return replace(volume, image=image)


class ProcessedVolumeWriter:
    '''
    Persist the image (int16 with NIfTI scaling, to save disk space), the brain mask and the transform.
    The image is written last through an atomic rename, so its presence marks a complete output.
    '''
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def image_path(self, scan_id: str) -> Path:
        return self.output_dir / f'{scan_id}_T1w_mni.nii.gz'

    def mask_path(self, scan_id: str) -> Path:
        return self.output_dir / f'{scan_id}_mask_mni.nii.gz'

    def transform_path(self, scan_id: str) -> Path:
        return self.output_dir / f'{scan_id}_to_mni.tfm'

    def exists(self, scan_id: str) -> bool:
        return self.image_path(scan_id).exists()

    def write(self, scan_id: str, volume: BrainVolume) -> Path:
        if volume.transform is not None:
            sitk.WriteTransform(volume.transform, str(self.transform_path(scan_id)))
        if volume.mask is not None:
            nib.save(sitk_to_nifti(volume.mask, np.uint8), str(self.mask_path(scan_id)))

        final_path = self.image_path(scan_id)
        partial_path = final_path.with_name(final_path.name.replace('.nii.gz', '.partial.nii.gz'))
        nib.save(sitk_to_nifti(volume.image, np.int16), str(partial_path))
        os.replace(partial_path, final_path)
        return final_path


class MRIPreprocessingPipeline:
    '''
    Run a sequence of volume steps over many scans.
    Resumable: scans whose output already exists are skipped. Failures are logged and do not stop the run.
    '''
    def __init__(
        self,
        reader: SitkVolumeReader,
        steps: list[VolumeStepInterface],
        writer: ProcessedVolumeWriter,
        failure_log_path: str,
    ):
        self.reader = reader
        self.steps = steps
        self.writer = writer
        self.failure_log_path = failure_log_path

    def run(self, scans: pd.DataFrame) -> None:
        pending = scans[~scans['ID'].map(self.writer.exists)]
        total = len(pending)
        print(f"--- Preprocessing: {len(scans) - total} already done, {total} to process ---")

        start = time.time()
        failures = 0
        for done, row in enumerate(pending.itertuples(index=False), start=1):
            step_name = 'read'
            try:
                volume = self.reader.read(row.Path)
                for step in self.steps:
                    step_name = type(step).__name__
                    volume = step.apply(volume)
                step_name = 'write'
                self.writer.write(row.ID, volume)
                status = '✓'
            except Exception as error:
                failures += 1
                status = f'✗ {step_name}: {error}'
                self._log_failure(row.ID, row.Path, step_name, error)

            elapsed = time.time() - start
            eta_min = elapsed / done * (total - done) / 60
            print(f"[{done}/{total}] {row.ID} {status} | {elapsed / done:.1f}s/scan | ETA {eta_min:.0f} min", flush=True)

        print(f"Preprocessing complete. {total - failures} succeeded, {failures} failed.")

    def _log_failure(self, scan_id: str, path: str, step_name: str, error: Exception) -> None:
        is_new = not os.path.exists(self.failure_log_path)
        with open(self.failure_log_path, 'a', newline='', encoding='utf-8') as log_file:
            writer = csv.writer(log_file)
            if is_new:
                writer.writerow(['ID', 'Path', 'Step', 'Error'])
            writer.writerow([scan_id, path, step_name, repr(error)])
