import os
import tempfile
import numpy as np
import nibabel as nib
import torch
from HD_BET.hd_bet_prediction import get_hdbet_predictor, hdbet_predict


class SkullStripper:
    """
    Remove skull from a 3D MRI volume using HD-BET 2.x.
    To swap the underlying model, subclass or inject a different implementation.
    """
    def __init__(self, use_gpu: bool = False, use_tta: bool = False, verbose: bool = False):
        device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        # Predictor is loaded once and reused for all volumes — avoids reloading weights each time
        self.predictor = get_hdbet_predictor(use_tta=use_tta, device=device, verbose=verbose)

    def strip(self, img: nib.Nifti1Image, patient_id: str, tmp_dir: str) -> nib.Nifti1Image:
        """
        Applies skull stripping to a NIfTI image using HD-BET 2.x API.
        Uses a shared temp directory to avoid redundant I/O.
        Returns a skull-stripped NIfTI image.
        """
        input_path = os.path.join(tmp_dir, f"{patient_id}_input.nii.gz")
        output_path = os.path.join(tmp_dir, f"{patient_id}_output.nii.gz")

        nib.save(img, input_path)
        hdbet_predict(
            input_file_or_folder=input_path,
            output_file_or_folder=output_path,
            predictor=self.predictor,
            keep_brain_mask=False,
            compute_brain_extracted_image=True
        )

        stripped_img = nib.load(output_path)
        # Copy data into memory before temp files are cleaned up
        stripped_data = stripped_img.get_fdata().copy()
        return nib.Nifti1Image(stripped_data, stripped_img.affine)


class CleanedMRIWriter:
    """
    Save a processed MRI volume to disk as a NIfTI file.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write(self, img: nib.Nifti1Image, filename: str) -> str:
        """
        Saves a NIfTI image to the output directory.
        Returns the full path of the saved file.
        """
        output_path = os.path.join(self.output_dir, filename)
        nib.save(img, output_path)
        return output_path


class SkullStrippingPipeline:
    """
    Coordinates the full skull-stripping workflow across all patients.
    Depends on SkullStripper and CleanedMRIWriter, not on their internals.
    Swap stripper or writer without modifying this class.
    """
    def __init__(self, stripper: SkullStripper, writer: CleanedMRIWriter):
        self.stripper = stripper
        self.writer = writer

    def run(self, dataframe) -> None:
        """
        Iterates over all rows in the dataframe, strips each MRI, and saves the result.
        Skips patients whose output file already exists, allowing safe re-runs.
        """
        total = len(dataframe)
        print(f"--- Starting Skull Stripping Pipeline ({total} volumes) ---")

        # Single temp directory shared across all patients — created once, cleaned once
        with tempfile.TemporaryDirectory() as tmp_dir:
            for idx, row in dataframe.iterrows():
                patient_id = row['ID']
                input_path = row['Path']
                output_filename = f"{patient_id}_stripped.nii.gz"
                output_full_path = os.path.join(self.writer.output_dir, output_filename)

                # Skip already processed files — safe to re-run after interruptions
                if os.path.exists(output_full_path):
                    print(f"[{idx+1}/{total}] ↷ {patient_id} — already processed, skipping.")
                    continue

                try:
                    img = nib.load(input_path)
                    stripped_img = self.stripper.strip(img, patient_id, tmp_dir)
                    output_path = self.writer.write(stripped_img, output_filename)
                    print(f"[{idx+1}/{total}] ✓ {patient_id} → {output_path}")

                except Exception as e:
                    print(f"[{idx+1}/{total}] ✗ {patient_id} — Error: {e}")
                    continue

        print("Skull stripping complete.")