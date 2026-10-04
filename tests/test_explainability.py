'''
Ground-truth check for the XAI pipeline.

A synthetic classifier whose class-1 evidence comes ONLY from the left hippocampus is explained with
every attribution method. If cropping, resizing, mapping back to MNI space, atlas alignment and
left/right handling are all correct, the left hippocampus must be ranked as the top region.

Run with:  python -m unittest discover tests
'''
import os
import tempfile
import unittest

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.brain_volume import array_to_nifti
from src.explainability import (
    GradCAM3D, IntegratedGradients3D, InputToMNIMapper, HarvardOxfordAtlasProvider,
    AtlasRegionSummarizer, ExplanationFigureRenderer, ExplanationService,
)
from src.preprocessing import CropResizeLoader

TEMPLATE_PATH = 'data/templates/mni152_brain_1mm.nii.gz'
TARGET_REGION = 'Left Hippocampus'


class RegionEvidenceModel(nn.Module):
    '''Toy classifier: the class-1 logit is the mean intensity inside a fixed region of the input grid.'''

    def __init__(self, region_mask: torch.Tensor):
        super().__init__()
        self.register_buffer('region_mask', region_mask)
        self.target_layer = nn.Conv3d(1, 1, kernel_size=1, bias=False)
        nn.init.ones_(self.target_layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        evidence = self.target_layer(x * self.region_mask).sum(dim=(1, 2, 3, 4)) / self.region_mask.sum()
        return torch.stack([-evidence, evidence], dim=1)


@unittest.skipUnless(os.path.exists(TEMPLATE_PATH), 'MNI template not materialized yet (run preprocessing once)')
class ExplanationLocalizationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.atlas_labels, cls.region_names = HarvardOxfordAtlasProvider(TEMPLATE_PATH).load()
        cls.loader = CropResizeLoader()
        cls.temp_dir = tempfile.TemporaryDirectory()

        # Input: the z-scored template brain, saved like a preprocessed scan
        template = nib.load(TEMPLATE_PATH)
        brain = np.asarray(template.get_fdata(dtype=np.float32))
        inside = brain > 0
        brain[inside] = (brain[inside] - brain[inside].mean()) / brain[inside].std()
        cls.scan_path = os.path.join(cls.temp_dir.name, 'template_T1w_mni.nii.gz')
        nib.save(array_to_nifti(brain, template.affine), cls.scan_path)

        # Ground-truth region, brought onto the network input grid exactly as the loader does
        region_id = next(key for key, name in cls.region_names.items() if name == TARGET_REGION)
        region_mni = (cls.atlas_labels == region_id).astype(np.float32)[cls.loader.crop]
        cls.region_input = F.interpolate(torch.from_numpy(np.ascontiguousarray(region_mni))[None, None],
                                         size=cls.loader.target_shape, mode='nearest')

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_every_method_finds_the_ground_truth_region(self):
        service = ExplanationService(
            model=RegionEvidenceModel(self.region_input),
            device=torch.device('cpu'),
            loader=self.loader,
            methods=[GradCAM3D(), IntegratedGradients3D(n_steps=8, internal_batch_size=2)],
            mapper=InputToMNIMapper(self.loader),
            summarizer=AtlasRegionSummarizer(self.atlas_labels, self.region_names),
            renderer=ExplanationFigureRenderer(),
            output_dir=self.temp_dir.name,
        )
        summary = service.explain('template', self.scan_path)

        for method_name, result in summary['methods'].items():
            with self.subTest(method=method_name):
                top = result['top_regions'][0]
                self.assertEqual(top['region'], TARGET_REGION)
                self.assertGreater(top['share'], 0.5)
                self.assertTrue(os.path.exists(result['map']))
                self.assertTrue(os.path.exists(result['figure']))


if __name__ == '__main__':
    unittest.main()
