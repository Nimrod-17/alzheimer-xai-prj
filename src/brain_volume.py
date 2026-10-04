from dataclasses import dataclass

import nibabel as nib
import numpy as np
import SimpleITK as sitk

# ITK works in LPS physical coordinates, NIfTI affines are expressed in RAS
LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


@dataclass
class BrainVolume:
    '''
    A 3D MRI volume travelling through the preprocessing pipeline.
    Each step returns a new BrainVolume, enriching it with a brain mask and/or a spatial transform.
    '''
    image: sitk.Image
    mask: sitk.Image | None = None
    transform: sitk.Transform | None = None


def sitk_to_nifti(image: sitk.Image, dtype: type = np.float32) -> nib.Nifti1Image:
    '''
    Convert a SimpleITK image to a nibabel image, preserving its physical geometry.
    Storage dtype handling is delegated to array_to_nifti.
    '''
    array = sitk.GetArrayFromImage(image).transpose(2, 1, 0)  # (z, y, x) -> (x, y, z)

    spacing = np.array(image.GetSpacing())
    direction = np.array(image.GetDirection()).reshape(3, 3)

    affine = np.eye(4)
    affine[:3, :3] = direction * spacing  # scale each column by its voxel size
    affine[:3, 3] = image.GetOrigin()

    return array_to_nifti(array, LPS_TO_RAS @ affine, dtype)


def array_to_nifti(array: np.ndarray, affine: np.ndarray, dtype: type = np.float32) -> nib.Nifti1Image:
    '''
    Wrap a (x, y, z) array and a RAS affine into a nibabel image.
    With an integer dtype and float data, values are stored with a symmetric scl_slope and zero intercept,
    so that background zeros stay exactly zero.
    '''
    if np.issubdtype(dtype, np.integer) and np.issubdtype(array.dtype, np.floating):
        max_abs = float(np.abs(array).max())
        slope = max_abs / np.iinfo(dtype).max if max_abs > 0 else 1.0
        nifti = nib.Nifti1Image(np.round(array / slope).astype(dtype), affine)
        nifti.header.set_slope_inter(slope, 0.0)
        return nifti

    nifti = nib.Nifti1Image(array, affine)
    nifti.set_data_dtype(dtype)
    return nifti
