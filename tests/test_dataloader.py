from src.data.dataloader import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders()

X, y = next(iter(train_loader))

print(X.shape)
print(y.shape)

print(X.dtype)
print(y.dtype)