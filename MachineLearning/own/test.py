from CBoW_optimized import model

if __name__ == "__main__":
    word = input("Introudce palabra: ")
    embedding = model.get_embedding(word)

    print(embedding)