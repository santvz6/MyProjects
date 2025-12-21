import cv2
import torch
import torch.nn as nn
import numpy as np

from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Subset, DataLoader

from encoders import MnistEncoder
from models import ConwayCNN
from torch_datasets import MnistDataset
from config import SEED, MNIST_DIR, DATA_DIR, MNIST_DATA_DIR, device
from logger_config import logger


class Main:

    def __init__(self):
        np.random.seed(SEED)


    def process_data(self, filename="mnist_data"):
        img_encoder = MnistEncoder(dataset_path=MNIST_DIR ,ouput_path=MNIST_DATA_DIR)

        encode_filename = filename + "_encoded"
        encode_file_path = MNIST_DATA_DIR / (encode_filename + ".npz")

        if not encode_file_path.exists():
            img_encoder.process_and_save(filename=encode_filename)

        dataset = MnistDataset(npz_path=encode_file_path, seed=SEED)

        if len(dataset) == 0:
            raise ValueError("Dataset vacío")
        else:
            logger.info("Encoded data succesfully loaded!")

        return dataset



    def train(self, dataset, test_size=0.2, n_splits=5, epochs=5, batch_size=64):
        logger.info(f"Using {device}...")
        labels = dataset.y # do not collapse RAM  np.ndarray (70_000, )
        idxs = np.arange(len(labels))

        # Visualization
        preview_count = {}
        
        # Train - Test Split
        trainval_idx, test_idx = train_test_split(idxs, test_size=test_size, stratify=labels, random_state=SEED)
        trainval_dataset = Subset(dataset, trainval_idx)
        test_dataset     = Subset(dataset, test_idx)
        logger.info(f"Train+Val size: {len(trainval_dataset)} | Test size: {len(test_dataset)}")

        trainval_labels = labels[trainval_idx]
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(trainval_idx)), trainval_labels)):
            logger.info(f"> FOLD {fold}")
 
            # Subsets
            train_dataset = Subset(trainval_dataset, train_idx)
            val_dataset   = Subset(trainval_dataset, val_idx)

            # DataLoaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

            # Model Configuration and Evaluation
            model = ConwayCNN().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            # Epochs loop
            for epoch in range(epochs):
                logger.info(f">> Epoch {epoch} | Batchs: {len(train_loader)}")
                model.train()
                train_loss = 0

                for batch, (images, labels) in enumerate(train_loader):
                    images, labels = images.to(device), labels.to(device)
                    
                    preview_count = self.__visualize(fold, epoch, images, labels, preview_count)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    if (batch % 100 == 0):
                        logger.info(f">>> Batch {batch} | Loss: {loss.item()}")

                # Fold Validation
                val_acc = self.__evaluate(model, val_loader)
                logger.info(f">> Fold {fold} | Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")


        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_acc = self.__evaluate(model, test_loader)
        logger.info(f"> FINAL TEST ACCURACY: {test_acc:.2f}%")

    
    def __evaluate(self, model, loader):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total
    
    def __visualize(self, fold, epoch, images, labels, preview_count, MAX_PREVIEW=5, SAVE_PREVIEW=True):
        if not (epoch == 0 and fold == 0):
            return preview_count

        if preview_count and all(v >= MAX_PREVIEW for v in preview_count.values()):
            return preview_count
        
        images_cpu = images.detach().cpu().numpy()
        labels_cpu = labels.detach().cpu().numpy()

        for i in range(len(images_cpu)):
            label = int(labels_cpu[i])
            preview_count.setdefault(label, 0)

            if preview_count[label] < MAX_PREVIEW:
                img = images_cpu[i, 0]  # (28, 28)

                save_path = (MNIST_DATA_DIR / "preview_conway"/ str(label) / f"sample_{preview_count[label]}.png")
                save_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_path),(img * 255).astype(np.uint8))
                preview_count[label] += 1
        
        return preview_count


if __name__ == "__main__":
    main = Main()
    dataset = main.process_data()
    main.train(dataset)