import torch.nn as nn
import torchvision.models as models

# ✅ Chargement d’un modèle MobileNetV2 pré-entraîné
def get_model(num_classes):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    # Remplace la dernière couche pour s’adapter au nombre de classes
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


# ✅ Modèle personnalisé basé sur MobileNetV2 sans poids pré-entraînés
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # On utilise les couches "features" de MobileNetV2 (sans les poids pré-entraînés)
        self.features = models.mobilenet_v2(weights=None).features
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])  # Global Average Pooling
        x = self.classifier(x)
        return x