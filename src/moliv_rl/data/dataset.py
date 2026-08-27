from torchvision.datasets import ImageFolder

from .transforms import get_train_transforms, get_val_transforms

# No need to get this too complicated. And no need to synthetic data too. In the future, maybe use Huggingface Datasets too.

train_dataset = ImageFolder(
    root="data/train",
    transform=get_train_transforms(image_size=64),
)

val_dataset = ImageFolder(
    root="data/val",
    transform=get_val_transforms(image_size=64),
)


