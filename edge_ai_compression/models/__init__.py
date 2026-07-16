"""Model builders for the compression framework (CIFAR-adapted torchvision nets)."""

from edge_ai_compression.models.efficientnet import efficientnet_b0_cifar
from edge_ai_compression.models.mobilenet import mobilenet_v2_cifar
from edge_ai_compression.models.resnet import resnet18_cifar
from edge_ai_compression.models.student_models import small_cnn_student

__all__ = [
    "resnet18_cifar",
    "mobilenet_v2_cifar",
    "efficientnet_b0_cifar",
    "small_cnn_student",
]
