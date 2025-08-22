from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_transforms(img_size=(128, 128)):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def get_dataloaders(data_dir, batch_size=32, img_size=(128, 128)):
    transform = get_data_transforms(img_size)

    train_ds = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
    val_ds = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)
    test_ds = datasets.ImageFolder(root=f"{data_dir}/test", transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader