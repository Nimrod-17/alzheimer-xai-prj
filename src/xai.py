import torch
import matplotlib.pyplot as plt
from captum.attr import LayerGradCam
import numpy as np

def generate_and_plot_gradcam(model, input_tensor, target_class, slice_idx=None):
    '''
    Generates a 3D Grad-CAM heatmap and plots a 2D slice overlay.
    '''
    # Model in evaluation mode
    model.eval()

    # Setup GradCAM
    target_layer = model.conv_block3
    gradcam = LayerGradCam(model, target_layer)

    # generate the 3D Heatmap
    attributions = gradcam.attribute(input_tensor, target=target_class) # target_class is 0 (healthy) or 1 (Alzheimer)

    # Upscale the heatmap
    import torch.nn.functional as F
    heatmap_3d = F.interpolate(attributions, size=input_tensor.shape[2:], mode='trilinear', align_corners=False)

    # Process tensors for plotting
    heatmap_3d = heatmap_3d.squeeze().cpu().detach().numpy()
    mri_3d = input_tensor.squeeze().cpu().detach().numpy()

    heatmap_3d = np.maximum(heatmap_3d, 0) # ReLU
    if np.max(heatmap_3d) > 0:
        heatmap_3d /= np.max(heatmap_3d)

    # Plotting
    if slice_idx is None:
        slice_idx = heatmap_3d.shape[0] // 2

    mri_slice = mri_3d[slice_idx, :, :]
    heatmap_slice = heatmap_3d[slice_idx, :, :]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    #  Original MRI
    axes[0].imshow(mri_slice, cmap='gray')
    axes[0].set_title('Original MRI (Slice {slice_idx})')
    axes[0].axis('off')

    # Raw Heatmap
    axes[1].imshow(heatmap_slice, cmap='jet')
    axes[1].set_title('AI Focus (Grad-CAM)')
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(mri_slice, cmap='gray')
    axes[2].imshow(heatmap_slice, cmap='jet', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()