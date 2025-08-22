# main.py

import sys
from pathlib import Path

from src.data import get_dataloaders
from src.model import get_model
from src.train_model import train_model

import torch

def main():
    print("🚀 Démarrage du pipeline d'entraînement")

    # 1️⃣ Définir les hyperparamètres
    data_dir = Path("data/processed/OpenAnimalTracks_split")
    batch_size = 32
    img_size = (128, 128)
    num_epochs = 10
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"📦 Utilisation du device : {device}")

    # 2️⃣ Charger les données
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size
    )

    print(f"📊 Nombre de classes : {len(class_names)}")
    print(f"🔠 Classes : {class_names}")

    # 3️⃣ Créer le modèle
    model = get_model(num_classes=len(class_names))
    model.to(device)

    # 4️⃣ Entraîner le modèle
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=num_epochs
    )

    import matplotlib.pyplot as plt

    # Tracer et sauvegarder la courbe d'apprentissage
    plt.figure(figsize=(10, 4))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_accuracy"], label="Validation Accuracy (%)")
    plt.title("Courbe d'apprentissage")
    plt.xlabel("Epochs")
    plt.ylabel("Valeurs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/learning_curves.png")
    plt.show()

    print("✅ Entraînement terminé ! Modèle sauvegardé.")

if __name__ == "__main__":
    main()