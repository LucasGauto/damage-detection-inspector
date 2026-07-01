from torchvision import transforms
from config import TARGET_SIZE_CLASSIFIER

def build_transform(target_size: int = TARGET_SIZE_CLASSIFIER) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    ) 