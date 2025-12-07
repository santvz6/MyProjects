import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class IrisData:
    def __init__(self, data_path: str) -> None:
        self.data_path = data_path
        

    def get_splited_data(self) -> tuple:
        iris = pd.read_csv(self.data_path, header= None)

        X = iris.iloc[:, :4].values.astype(float)
        y_str = iris.iloc[:, -1].values.astype(str)

        self.labels = np.unique(y_str)
        label_to_int = {label: i for i, label in enumerate(self.labels)}
        y = np.array([label_to_int[label] for label in y_str])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,        # mantiene la proporción de clases
            random_state=42
        )
        
        return X_train, X_test, y_train, y_test


iris = IrisData(data_path= "chapters/iris/iris.data")