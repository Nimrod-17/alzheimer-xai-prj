import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import LayerGradCam


def generate_and_plot_gradcam(model, input_tensor: torch.Tensor, target_class: int, slice_idx: int = None) -> None:
    """
    Generates a 3D Grad-CAM heatmap and plots a 2D axial slice overlay.

    Args:
        model:          Trained Alzheimer3DCNN instance.
        input_tensor:   Input MRI tensor of shape (1, 1, D, H, W).
        target_class:   Class index to explain — 0 (healthy) or 1 (Alzheimer's).
        slice_idx:      Axial slice index to visualize. Defaults to the middle slice.
    """
    model.eval()

    # Attach Grad-CAM to the last convolutional block
    gradcam = LayerGradCam(model, model.conv_block3)
    attributions = gradcam.attribute(input_tensor, target=target_class)

    # Upsample heatmap to match the original input spatial dimensions
    heatmap_3d = F.interpolate(
        attributions,
        size=input_tensor.shape[2:],
        mode='trilinear',
        align_corners=False
    )

    # Convert tensors to numpy arrays for plotting
    heatmap_3d = heatmap_3d.squeeze().cpu().detach().numpy()
    mri_3d = input_tensor.squeeze().cpu().detach().numpy()

    # Apply ReLU and normalize heatmap to [0, 1]
    heatmap_3d = np.maximum(heatmap_3d, 0)
    if np.max(heatmap_3d) > 0:
        heatmap_3d /= np.max(heatmap_3d)

    # Default to the middle axial slice if none is specified
    if slice_idx is None:
        slice_idx = heatmap_3d.shape[0] // 2

    mri_slice = mri_3d[slice_idx, :, :]
    heatmap_slice = heatmap_3d[slice_idx, :, :]

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(mri_slice, cmap='gray')
    axes[0].set_title(f'Original MRI (Slice {slice_idx})')
    axes[0].axis('off')

    axes[1].imshow(heatmap_slice, cmap='jet')
    axes[1].set_title('AI Focus (Grad-CAM)')
    axes[1].axis('off')

    axes[2].imshow(mri_slice, cmap='gray')
    axes[2].imshow(heatmap_slice, cmap='jet', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()