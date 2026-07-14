import os
import torch
from src.model import Alzheimer3DCNN
from src.dataset import MRIDataset
from src.preprocessing import ZScoreNormalizer # Usiamo quello nuovo!
from src.data_loader import ExcelClinicalParser, MRIFileResolver, MRIDataLoader
from src.xai import generate_and_plot_gradcam

# 1. Carichiamo l'architettura vuota e ci inseriamo i "ricordi" dell'Epoca 27
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Alzheimer3DCNN().to(device)
model.load_state_dict(torch.load('models/best_model.pth', weights_only=True))
model.eval()

# 2. Ripeschiamo un paziente (usiamo lo stesso DataLoader del main)
parser = ExcelClinicalParser(excel_path='data/Demographic and Clinical Data/oasis_cross-sectional.xlsx')
resolver = MRIFileResolver(base_data_dir='data/cleaned_data', search_pattern="{id}_stripped.nii.gz")
df = MRIDataLoader(parser, resolver).load_data()

# Prendiamo un paziente malato (Label == 1) per vedere cosa nota l'IA
malati_df = df[df['Label'] == 1].reset_index(drop=True)
dataset = MRIDataset(dataframe=malati_df, preprocessor=ZScoreNormalizer())

# Estraiamo il primo paziente malato (Aggiungiamo la dimensione Batch)
img_tensor, label_tensor = dataset[0]
img_tensor = img_tensor.unsqueeze(0).to(device)

print("Generazione mappa di calore in corso...")
# 3. Evochiamo il Grad-CAM!
generate_and_plot_gradcam(model, img_tensor, target_class=1)