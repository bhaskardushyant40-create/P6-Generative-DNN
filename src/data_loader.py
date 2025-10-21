import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_dataloaders(batch_size=64, root='./data'):
    """
    Loads MNIST training and testing data.
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def get_fashion_mnist_dataloaders(batch_size=64, root='./data'):
    """
    Loads Fashion MNIST data. Used for the first transfer learning task.
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return test_loader 

def get_faces_dataloader(batch_size=64, root='./data'):
    """
    Loads a Face dataset (e.g., CelebA) which requires different transforms.
    NOTE: torchvision.datasets.CelebA might require manual download/setup.
    This uses the default PyTorch CelebA class.
    """
    img_size = 64 
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])

    try:
        face_dataset = datasets.CelebA(
            root=root,
            split='train',
            transform=transform,
            download=False 
        )
        face_loader = DataLoader(face_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        return face_loader
    except RuntimeError:
        print("CelebA download/setup failed. Ensure the dataset is downloaded manually or use a different face dataset.")
        return None
