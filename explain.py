import argparse
import json

import torch

from src.explainability import (
    GradCAM3D, IntegratedGradients3D, InputToMNIMapper, HarvardOxfordAtlasProvider,
    AtlasRegionSummarizer, ExplanationFigureRenderer, ExplanationService,
)
from src.factories import build_data_module, build_model, INPUT_SHAPE
from src.preprocessing import CropResizeLoader

# --- Configuration ---
MODEL_PATH = 'models/oasis3_resnet3d.pth'
METADATA_PATH = 'models/oasis3_resnet3d.json'
TEMPLATE_PATH = 'data/templates/mni152_brain_1mm.nii.gz'
OUTPUT_DIR = 'results/explanations'


def main():
    parser = argparse.ArgumentParser(description='Generate XAI explanations for preprocessed OASIS-3 scans.')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--ids', nargs='*', help='Explain only these scan IDs (within the split).')
    parser.add_argument('--limit', type=int, default=None, help='Explain at most N scans.')
    parser.add_argument('--ig-steps', type=int, default=32, help='Integrated Gradients interpolation steps.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location=device))
    with open(METADATA_PATH, encoding='utf-8') as metadata_file:
        threshold = json.load(metadata_file)['threshold']

    # Same scans the evaluation uses: one fixed scan per subject
    data = build_data_module(augment=False)
    data.setup()
    frame = data.frames[args.split]
    if args.ids:
        frame = frame[frame['ID'].isin(args.ids)]
    if args.limit is not None:
        frame = frame.head(args.limit)

    loader = CropResizeLoader(target_shape=INPUT_SHAPE)
    atlas_labels, region_names = HarvardOxfordAtlasProvider(TEMPLATE_PATH).load()
    service = ExplanationService(
        model=model,
        device=device,
        loader=loader,
        methods=[GradCAM3D(), IntegratedGradients3D(n_steps=args.ig_steps)],
        mapper=InputToMNIMapper(loader),
        summarizer=AtlasRegionSummarizer(atlas_labels, region_names),
        renderer=ExplanationFigureRenderer(),
        output_dir=OUTPUT_DIR,
        decision_threshold=threshold,
    )

    print(f"--- Explaining {len(frame)} scans from the {args.split} split ---")
    for row in frame.itertuples(index=False):
        summary = service.explain(row.ID, row.Path, label=int(row.Label))
        top = summary['methods']['gradcam']['top_regions'][:3]
        regions = ', '.join(f"{region['region']} (x{region['enrichment']:.1f})" for region in top)
        print(f"{row.ID} | label {row.Label} | P = {summary['probability']:.2f} | Grad-CAM top: {regions}", flush=True)

    print(f"Explanations saved in {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
