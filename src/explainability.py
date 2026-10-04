import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients, LayerGradCam
from scipy.ndimage import gaussian_filter

from src.brain_volume import array_to_nifti
from src.interfaces import AttributionMethodInterface
from src.preprocessing import CropResizeLoader


# --- Attribution methods (network input grid) ---

class GradCAM3D(AttributionMethodInterface):
    '''
    Grad-CAM on the model's last convolutional stage (model.target_layer), upsampled to the input grid.
    Coarse (8x10x8 before upsampling) but class-discriminative and robust: shows WHICH REGIONS drive the score.
    '''
    name = 'gradcam'

    def attribute(self, model, volume: torch.Tensor, target: int) -> np.ndarray:
        cam = LayerGradCam(model, model.target_layer).attribute(volume, target=target, relu_attributions=True)
        cam = F.interpolate(cam, size=volume.shape[2:], mode='trilinear', align_corners=False)
        return cam[0, 0].detach().cpu().numpy()


class IntegratedGradients3D(AttributionMethodInterface):
    '''
    Integrated Gradients from a zero baseline (z-score 0, i.e. both the background and the mean brain intensity).
    Voxel-level and axiomatic (attributions sum to the logit difference from the baseline):
    shows WHICH VOXELS, with sign — positive pushes towards the target class.
    '''
    name = 'integrated_gradients'

    def __init__(self, n_steps: int = 32, internal_batch_size: int = 4):
        self.n_steps = n_steps
        self.internal_batch_size = internal_batch_size

    def attribute(self, model, volume: torch.Tensor, target: int) -> np.ndarray:
        attributions = IntegratedGradients(model).attribute(
            volume, baselines=torch.zeros_like(volume), target=target,
            n_steps=self.n_steps, internal_batch_size=self.internal_batch_size,
        )
        return attributions[0, 0].detach().cpu().numpy()


# --- Spatial mapping ---

class InputToMNIMapper:
    '''
    Invert CropResizeLoader: resize a map from the network input grid back to the crop box
    and paste it into the full 1 mm MNI grid, so it overlays the preprocessed T1 and the atlas.
    '''
    def __init__(self, loader: CropResizeLoader, mni_shape: tuple[int, int, int] = (197, 233, 189)):
        self.crop = loader.crop
        self.mni_shape = mni_shape

    def to_mni(self, array: np.ndarray) -> np.ndarray:
        crop_shape = tuple(s.stop - s.start for s in self.crop)
        tensor = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))[None, None]
        resized = F.interpolate(tensor, size=crop_shape, mode='trilinear', align_corners=False)[0, 0].numpy()

        full = np.zeros(self.mni_shape, dtype=np.float32)
        full[self.crop] = resized
        return full


# --- Anatomical summary ---

class HarvardOxfordAtlasProvider:
    '''
    Harvard-Oxford cortical (left/right split) + subcortical atlas, resampled once onto the MNI template grid.
    Generic subcortical labels (cerebral white matter / cortex) are dropped: cortex is covered by the cortical atlas.
    Note: the atlas lives in FSL's MNI152 space, the template is ICBM 2009; the small mismatch is
    acceptable for region-level summaries.
    '''
    SUBCORTICAL_OFFSET = 100
    DROPPED_SUBCORTICAL = ('Cerebral White Matter', 'Cerebral Cortex')

    def __init__(self, template_path: str, cache_dir: str = 'data/atlases'):
        self.template_path = template_path
        self.cache_dir = Path(cache_dir)

    def load(self) -> tuple[np.ndarray, dict[int, str]]:
        atlas_path = self.cache_dir / 'harvard_oxford_on_template.nii.gz'
        names_path = self.cache_dir / 'harvard_oxford_on_template.json'
        if not (atlas_path.exists() and names_path.exists()):
            self._build(atlas_path, names_path)

        labels = np.asarray(nib.load(str(atlas_path)).dataobj).astype(np.int16)
        with open(names_path, encoding='utf-8') as names_file:
            names = {int(key): value for key, value in json.load(names_file).items()}
        return labels, names

    def _build(self, atlas_path: Path, names_path: Path) -> None:
        from nilearn import datasets, image

        template = nib.load(self.template_path)
        cortical = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm', data_dir=str(self.cache_dir), symmetric_split=True)
        subcortical = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-1mm', data_dir=str(self.cache_dir))

        def on_template(atlas) -> np.ndarray:
            resampled = image.resample_to_img(atlas.maps, template, interpolation='nearest',
                                              force_resample=True, copy_header=True)
            return np.asarray(resampled.dataobj).astype(np.int16)

        labels = on_template(cortical)
        names = {index: name for index, name in enumerate(cortical.labels) if index > 0}

        subcortical_labels = on_template(subcortical)
        for index, name in enumerate(subcortical.labels):
            if index == 0 or any(dropped in name for dropped in self.DROPPED_SUBCORTICAL):
                continue
            # Subcortical structures are more specific than the cortical maxprob labels: they take precedence
            labels[subcortical_labels == index] = self.SUBCORTICAL_OFFSET + index
            names[self.SUBCORTICAL_OFFSET + index] = name

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(labels, template.affine), str(atlas_path))
        with open(names_path, 'w', encoding='utf-8') as names_file:
            json.dump(names, names_file, indent=2)


class AtlasRegionSummarizer:
    '''
    Summarize a positive-evidence map by anatomical region:
    - share: fraction of the total positive attribution falling in the region;
    - enrichment: region mean / whole-brain mean (>1 = the model focuses there more than average).
    Regions are ranked by enrichment, ignoring very small ones whose means are unstable.
    '''
    def __init__(self, atlas_labels: np.ndarray, region_names: dict[int, str], min_voxels: int = 500):
        self.atlas_labels = atlas_labels
        self.region_names = region_names
        self.min_voxels = min_voxels

    def summarize(self, attribution: np.ndarray, brain_mask: np.ndarray, top_k: int = 10) -> list[dict]:
        positive = np.clip(attribution, 0, None) * brain_mask
        total = positive.sum()
        brain_mean = positive[brain_mask].mean() if brain_mask.any() else 0.0
        if total <= 0 or brain_mean <= 0:
            return []

        regions = []
        for label, name in self.region_names.items():
            region = (self.atlas_labels == label) & brain_mask
            voxels = int(region.sum())
            if voxels < self.min_voxels:
                continue
            regions.append({
                'region': name,
                'enrichment': float(positive[region].mean() / brain_mean),
                'share': float(positive[region].sum() / total),
                'voxels': voxels,
            })

        return sorted(regions, key=lambda region: region['enrichment'], reverse=True)[:top_k]


# --- Visualization ---

class ExplanationFigureRenderer:
    '''
    Orthogonal slices through the peak of positive evidence, attribution overlaid on the patient's T1.
    Display only: the map is lightly smoothed and normalized by a high percentile.
    '''
    def __init__(self, smoothing_sigma: float = 2.0, display_percentile: float = 99.5, min_alpha_level: float = 0.25):
        self.smoothing_sigma = smoothing_sigma
        self.display_percentile = display_percentile
        self.min_alpha_level = min_alpha_level

    def render(self, t1: np.ndarray, attribution: np.ndarray, title: str, output_path: Path) -> None:
        positive = gaussian_filter(np.clip(attribution, 0, None), self.smoothing_sigma)
        scale = np.percentile(positive[positive > 0], self.display_percentile) if (positive > 0).any() else 1.0
        overlay = np.clip(positive / max(scale, 1e-12), 0, 1)
        peak = np.unravel_index(np.argmax(overlay), overlay.shape)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
        planes = [('Sagittal', 0), ('Coronal', 1), ('Axial', 2)]
        for axis, (plane_name, plane_axis) in zip(axes, planes):
            t1_slice = np.rot90(np.take(t1, peak[plane_axis], axis=plane_axis))
            map_slice = np.rot90(np.take(overlay, peak[plane_axis], axis=plane_axis))
            axis.imshow(t1_slice, cmap='gray')
            axis.imshow(np.ma.masked_less(map_slice, self.min_alpha_level), cmap='hot', vmin=0, vmax=1, alpha=0.6)
            axis.set_title(f'{plane_name} (slice {peak[plane_axis]})', fontsize=9)
            axis.axis('off')
            if plane_axis != 0:
                # Neurological convention: the patient's left is shown on the left
                for x_position, side in ((0.02, 'L'), (0.98, 'R')):
                    axis.text(x_position, 0.5, side, transform=axis.transAxes, color='yellow',
                              fontsize=11, fontweight='bold', ha='center', va='center')

        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(output_path, dpi=90)
        plt.close(fig)


# --- Orchestration ---

class ExplanationService:
    '''
    Explain one preprocessed scan: predict, run every attribution method, map results to MNI space,
    summarize them by anatomical region and persist maps (NIfTI), figures (PNG) and a JSON summary
    that the clinical dashboard can consume.
    '''
    def __init__(self, model, device: torch.device, loader: CropResizeLoader,
                 methods: list[AttributionMethodInterface], mapper: InputToMNIMapper,
                 summarizer: AtlasRegionSummarizer, renderer: ExplanationFigureRenderer,
                 output_dir: str, target_class: int = 1, decision_threshold: float = 0.5):
        self.model = model.eval()
        self.device = device
        self.loader = loader
        self.methods = methods
        self.mapper = mapper
        self.summarizer = summarizer
        self.renderer = renderer
        self.output_dir = Path(output_dir)
        self.target_class = target_class
        self.decision_threshold = decision_threshold

    def explain(self, scan_id: str, image_path: str, label: int | None = None) -> dict:
        scan_dir = self.output_dir / scan_id
        scan_dir.mkdir(parents=True, exist_ok=True)

        mni_image = nib.load(image_path)
        t1_mni = np.asarray(mni_image.get_fdata(dtype=np.float32))
        brain_mni = t1_mni != 0

        volume = torch.from_numpy(self.loader.preprocess(image_path))[None, None].to(self.device)
        brain_input = (volume[0, 0] != 0).cpu().numpy()

        with torch.no_grad():
            probability = torch.softmax(self.model(volume).float(), dim=1)[0, self.target_class].item()

        summary = {
            'scan_id': scan_id,
            'true_label': label,
            'probability': probability,
            'decision_threshold': self.decision_threshold,
            'predicted_label': int(probability >= self.decision_threshold),
            'explained_class': self.target_class,
            'methods': {},
        }

        for method in self.methods:
            attribution = self.mapper.to_mni(method.attribute(self.model, volume, self.target_class) * brain_input)
            attribution *= brain_mni

            map_path = scan_dir / f'{method.name}_mni.nii.gz'
            nib.save(array_to_nifti(attribution, mni_image.affine, np.int16), str(map_path))

            figure_path = scan_dir / f'{method.name}.png'
            title = f'{scan_id} | {method.name} | P(class {self.target_class}) = {probability:.2f}'
            self.renderer.render(t1_mni, attribution, title, figure_path)

            summary['methods'][method.name] = {
                'map': str(map_path),
                'figure': str(figure_path),
                'top_regions': self.summarizer.summarize(attribution, brain_mni),
            }

        with open(scan_dir / 'explanation.json', 'w', encoding='utf-8') as summary_file:
            json.dump(summary, summary_file, indent=2)
        return summary
