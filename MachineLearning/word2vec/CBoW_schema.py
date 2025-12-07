# Continuous Bag of Words 
# One of the Word2Vec FrameWork types 
import numpy as np
import re

class CBoW:

    def __init__(self, corpus, seed=1):
        
        self.seed: int          = seed
        self.rgen               = np.random.RandomState(seed=self.seed) 

        
        self.corpus: str        = re.sub(r'[^\w\s]', "", corpus.lower())
        self.tokens: list[str]  = self.corpus.split()
        self.word2idx: dict     = {word: i for i, word in enumerate(set(self.tokens))}


    def _one_hot(self, idx,  n):
        v = np.zeros(n)
        v[idx] = 1
        return v

    def _softmax(self, X):
        e = np.exp(X - np.max(X))
        return e / np.sum(e)

    def __generate_training_data(self, ngram_range= 2):
        """ Generamos los datos de entrenamiento en una lista de tuplas[contexto, target]"""
        self.training_data_: list[tuple] = []

        for i, word in enumerate(self.tokens):
            target = self.word2idx[word]    # trabajamos con los indices
            context = []                    # para cada token incluiremos ngram_range plabaras de ambos lados (sus indices realmente)

            # queremos (ngram_extra x 2) de contexto 
            # Recorremos +- range(1, ngram_range+1)
            for j in range(1, ngram_range + 1): 
                
                # izquierda
                if ((i - j) >= 0): 
                    context.append(self.word2idx[self.tokens[i - j]])
                    
                # derecha    
                if ((i + j) < len(self.tokens)):
                    context.append(self.word2idx[self.tokens[i + j]])
                
            if context:
                self.training_data_.append((context, target)) # guardamos el contexto del target (en índices)

    def fit(self, lr=0.1, epochs=1000, embedding_dim=10):
        """ Seguimos el modelo de perceptron visto en el Chapter 2"""
        self.__generate_training_data()

        # Nuestra matriz de pesos tendrá el tamaño de
        # Vocabulario |V| = len(self.word2idx) filas y
        # Dimensiones = embedding_dim columnas
        self.W1_ = self.rgen.normal(loc=0, scale= 0.01, size=(len(self.word2idx), embedding_dim))  # |V| x dim
        
        # Nuestra segunda cpada del perceptron matriz servirá para predecir cuál es la palabra target
        # Dimensiones = embedding_dim filas y
        # Vocabulario |V| = len(self.word2idx) columnas
        self.W2_ = self.rgen.normal(loc=0, scale= 0.01, size=(embedding_dim, len(self.word2idx)))  # dim x |V|


        for epoch in range(epochs):
            loss_epoch = 0

            for context_idxs, target_idx in self.training_data_:
                
                # En documentación externa suelen nombrar "x" como "h"
                X_context = [self._one_hot(context_idx, len(self.word2idx)) for context_idx in context_idxs] # len(context_idxs) x |V|

                updates = []
                for x_context in X_context:
                    updates.append(x_context @ self.W1_) # @: x1w1_ + x2w2 + ... + xnwn

                # Promediamos el valor de las filas (es decir el promedio de cada embedding para la misma palabra)
                # Mantenemos el número de columnas (palabras) pero acabamos con un único embedding promediado:
                x = np.mean(updates, axis=0) # vector embedding de contexto 1 x dim

                y_pred  = self.predict(x)
                y       = self._one_hot(target_idx, len(self.word2idx)) # posición real del target

                # Simplemente tendremos en cuenta aquel caso donde tengamos 1 en el one hot (para los demás (0 * ... = 0))
                # Cuánto mayor sea y_pred mejor minimizamos "loss" 
                loss = -np.sum(y * np.log(y_pred + 1e-9))
                loss_epoch += loss

                # Backpropagate (actualización de pesos al igual que en un Perceptrón)

                # Gradiente de salida (predecido vs real)
                e = y_pred - y          # Modificamos proporcionalmente aquellos que no hemos predecido bien. 1 x |V|
                dW2 = np.outer(x, e)    # Equivale a hacer: np.transpose(x) * e (np.outer aplana arrays (1, n) -> (n,))
                self.W2_ += dW2 * lr

                
                # Gradiente de embeddings (respecto a x)
                dx = e @ self.W2_.T # @: e1w1 + e2w2 + ... + enwn

                for context_idx in context_idxs:
                    self.W1_[context_idx] -= (dx / len(context_idxs)) * lr

            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {(loss_epoch/len(self.training_data_)):.4f}")


    def predict(self, x):
        u = x @ self.W2_            # Puntuación: cuánto cree la red que u[i] es el target
        y_pred = self._softmax(u)   # Probabilidad: probabilidad de que u[i] sea el target

        return y_pred


if __name__ == "__main__":
    from datasets import load_dataset


    dataset = load_dataset("text", data_files="own/dataset/spanish_corpora/all_wikis.txt")

    # Convertir a un solo string
    corpus = " ".join(dataset["train"]["text"])

    cbow = CBoW(corpus)
    cbow.fit(lr= 0.01, epochs=1000, embedding_dim= 20)
