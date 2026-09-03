import torch
import torch.nn as nn
from typing import Optional
import timm


class BreedClassifier(nn.Module):
    """
    Transfer Learning Model for Dog & Cat Breed Classification.
    """
    def __init__(self,
                 model_name: str = 'efficientnet_b0',
                 num_classes: int = 104,
                 pretrained: bool = True,
                 dropout_rate: float = 0.2):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Create backbone using timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout_rate
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self):
        """
        Freeze backbone parameters for initial feature extraction.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze classifier head
        classifier = self.backbone.get_classifier()
        if isinstance(classifier, nn.Module):
            for param in classifier.parameters():
                param.requires_grad = True
        elif isinstance(classifier, nn.Parameter):
            classifier.requires_grad = True

    def unfreeze_backbone(self):
        """
        Unfreeze all parameters for full fine-tuning.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True


def build_model(model_name: str = 'efficientnet_b0',
                num_classes: int = 104,
                pretrained: bool = True,
                dropout_rate: float = 0.2) -> BreedClassifier:
    """
    Factory function to instantiate the BreedClassifier.
    """
    return BreedClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )
