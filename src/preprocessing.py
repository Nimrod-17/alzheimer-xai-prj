import numpy as np
import nibabel as nib

def load_and_preprocess_image(file_path):
    """
    Loads a 3D NIfTI/Analyze file and normalizes voxel values 
    to a scale from 0.0 to 1.0 (Min-Max Scaling).
    """
    # 1. Load the image object
    img = nib.load(file_path)
    
    # 2. Extract the raw numerical data array
    img_data = img.get_fdata()
    
    # 3. Remove single-dimensional entries from the shape (e.g., [176, 208, 176, 1] becomes [176, 208, 176])
    img_data = np.squeeze(img_data)
    
    # 4. Min-Max Normalization
    min_val = np.min(img_data)
    max_val = np.max(img_data)
    
    # Avoid division by zero for completely black images
    if max_val - min_val > 0:
        img_normalized = (img_data - min_val) / (max_val - min_val)
    else:
        img_normalized = img_data
        
    return img_normalized