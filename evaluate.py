import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split


from src.data_loader import ExcelClinicalParser, MRIFileResolver, MRIDataLoader
from src.preprocessing import ZScoreNormalizer
from src.dataset import MRIDataset
from src.model import Alzheimer3DCNN

# --- CONFIGURAZIONE ---
EXCEL_PATH = 'data/Demographic and Clinical Data/oasis_cross-sectional.xlsx'
DATA_DIR = 'data/cleaned_data'
MODEL_PATH = 'models/best_model.pth'
BATCH_SIZE = 4
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

def main():
    print("--- 🩺 Avvio Valutazione Clinica Modello ---")
    
    # 1. Preparazione dell'hardware e del modello
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Alzheimer3DCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval() # Spegniamo il Dropout per l'esame!
    
    # 2. Caricamento dei dati
    print("Caricamento pazienti...")
    parser = ExcelClinicalParser(excel_path=EXCEL_PATH)
    resolver = MRIFileResolver(base_data_dir=DATA_DIR, search_pattern="{id}_stripped.nii.gz")
    df = MRIDataLoader(parser, resolver).load_data()
    
    dataset = MRIDataset(dataframe=df, preprocessor=ZScoreNormalizer())

    total_size = len(dataset)
    test_size = int(total_size * TEST_SPLIT)
    val_size = int(total_size * VAL_SPLIT)
    train_size = total_size - test_size - val_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)

    _, _, test_ds = random_split(
        dataset, 
        [train_size, val_size, test_size], 
        generator=generator
    )

    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Contatori per la Matrice di Confusione
    TP = 0  # True Positive (Malato previsto Malato)
    TN = 0  # True Negative (Sano previsto Sano)
    FP = 0  # False Positive (Sano previsto Malato - Falso Allarme)
    FN = 0  # False Negative (Malato previsto Sano - Caso Perso!)
    
    print("Analisi delle risonanze in corso...\n")
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Troviamo la classe prevista (0 o 1)
            probability = torch.softmax(outputs, dim=1)

            prob_alz = probability[:,1]
            soglia = 0.40
            predicted = (prob_alz >= soglia).int()
            
            for i in range(len(labels)):
                vero = labels[i].item()
                previsto = predicted[i].item()
                
                if vero == 1 and previsto == 1:
                    TP += 1
                elif vero == 0 and previsto == 0:
                    TN += 1
                elif vero == 0 and previsto == 1:
                    FP += 1
                elif vero == 1 and previsto == 0:
                    FN += 1
                    
    # 4. Calcolo delle Metriche Mediche
    totale = TP + TN + FP + FN
    accuratezza = (TP + TN) / totale if totale > 0 else 0
    sensibilita = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificita = TN / (TN + FP) if (TN + FP) > 0 else 0
    
    # 5. Stampa del Referto
    print("="*40)
    print(" 🏥 REFERTO UFFICIALE DELL'IA 🏥")
    print("="*40)
    print(f"Pazienti Totali Analizzati : {totale}")
    print("-" * 40)
    print(f"✅ Accuratezza Globale    : {accuratezza * 100:.2f}%")
    print(f"🔴 Sensibilità (Malati)   : {sensibilita * 100:.2f}%")
    print(f"🟢 Specificità (Sani)     : {specificita * 100:.2f}%")
    print("-" * 40)
    print("MATRICE DI CONFUSIONE:")
    print(f"  Vero Positivo (TP): {TP} | Falso Positivo (FP): {FP}")
    print(f"  Falso Negativo(FN): {FN} | Vero Negativo (TN): {TN}")
    print("="*40)

if __name__ == '__main__':
    main()