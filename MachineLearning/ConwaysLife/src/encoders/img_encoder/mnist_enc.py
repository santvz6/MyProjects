import cv2
import numpy as np

from pathlib import Path
from logger_config import logger
from config import MNIST_DATA_DIR

class MnistEncoder:
    def __init__(self, dataset_path: str|Path, ouput_path:str|Path):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(ouput_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
    def __binarize(self, img_path: Path):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None: return None
        
        # Binarization
        _, thresh = cv2.threshold(img, 255//2, 1, cv2.THRESH_BINARY)
        return thresh

    def process_and_save(self, filename:str):
        images, labels = [], []
        preview_count = {}
        max_preview_per_class = 10
  
        for img_path in self.dataset_path.rglob("*.png"):
            binary_2d = self.__binarize(img_path)
            if binary_2d is not None:
                images.append(binary_2d)
                labels.append(int(img_path.parent.name))

                save_path = MNIST_DATA_DIR / "preview" / img_path.parent.name / f"{img_path.stem}_label{img_path.parent.name}.png"
                if (len(images) < 10):
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(save_path, binary_2d*255)

        X = np.array(images) # shape: (num_imgs, 28, 28)
        y = np.array(labels)
        
        np.savez_compressed(self.output_path / filename, x=X, y=y)
        logger.info(f"> {MnistEncoder.__name__}: encoded data successfully saved!")
        
        images.clear()
        labels.clear()

