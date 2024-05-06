from transformers import BertTokenizer, BertModel
import torch
import pickle
import string

class VectorDatabase:
    """
    A class representing a vector database.

    Attributes:
        model_name (str): The name of the model (e.g. BERT) to use for vectorization.
        model (BertModel): The BERT model used for vectorization.
        tokenizer (BertTokenizer): The BERT tokenizer used for tokenization.
        vectors (dict): A dictionary to store the vectors.

    Methods:
        __init__(self, model_name=None): Initializes a VectorDatabase object.
        preprocess_text(self, text): Preprocesses the input text.
        add_vector(self, key, text): Adds a vector to the database.
        vectorize_text(self, text): Vectorizes the input text.
        query(self, text, k=10): Performs a query on the database.
        cosine_similarity(vector1, vector2): Computes the cosine similarity between two vectors.
        save(self, filename): Saves the vectors to a file.
        load(self, filename): Loads the vectors from a file.
    """

    def __init__(self, model_name=None):
        """
        Initializes a VectorDatabase object.

        Args:
            model_name (str, optional): The name of the BERT model to use for vectorization.
        """
        self.model = BertModel.from_pretrained('bert-base-uncased').eval()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.vectors = {}

    def preprocess_text(self, text):
        """
        Preprocesses the input text.

        Args:
            text (str): The input text to preprocess.

        Returns:
            list: The preprocessed tokens.
        """
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Convert to lowercase and split into tokens
        tokens = text.lower().split()
        return tokens

    def add_vector(self, key, text):
        """
        Adds a vector to the database.

        Args:
            key (str): The key to associate with the vector.
            text (str): The text to vectorize and add to the database.
        """
        vector_emb = self.vectorize_text(text)
        self.vectors[key] = vector_emb

    def vectorize_text(self, text):
        """
        Vectorizes the input text.

        Args:
            text (str): The text to vectorize.

        Returns:
            numpy.ndarray: The vectorized representation of the text.
        """
        inputs = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        pooled_embedding = outputs.last_hidden_state.mean(dim=1)[0]  
        return pooled_embedding.numpy()

    def query(self, text, k=10):
        """
        Performs a query on the database.

        Args:
            text (str): The query text.
            k (int, optional): The number of most similar vectors to return.

        Returns:
            list: A list of tuples containing the key and similarity score (as tensors) of the most similar vectors.
        """
        inputs = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        query_embedding = outputs.last_hidden_state.mean(dim=1)[0]
        similarities = [(key, self.cosine_similarity(query_embedding.numpy(), vector)) for key, vector in self.vectors.items()]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    @staticmethod
    def cosine_similarity(vector1, vector2):
        """
        Computes the cosine similarity between two vectors.

        Args:
            vector1 (numpy.ndarray): The first vector.
            vector2 (numpy.ndarray): The second vector.

        Returns:
            float: The cosine similarity between the two vectors.
        """
        dot_product = torch.dot(torch.tensor(vector1), torch.tensor(vector2))
        norm1 = torch.norm(torch.tensor(vector1))
        norm2 = torch.norm(torch.tensor(vector2))
        return dot_product / (norm1 * norm2)
    
    def save(self, filename):
        """
        Saves the vectors to a file.

        Args:
            filename (str): The name of the file to save the vectors to.
        """
        with open(filename, 'wb') as file:
            pickle.dump(self.vectors, file)

    def load(self, filename):
        """
        Loads the vectors from a file.

        Args:
            filename (str): The name of the file to load the vectors from.
        """
        with open(filename, 'rb') as file:
            self.vectors = pickle.load(file)


if __name__ == '__main__':
    db = VectorDatabase()
    db.add_vector('hello', 'Hello, world!')
    db.add_vector('goodbye', 'Goodbye, world!')
    db.save('vectors.pkl')

    db2 = VectorDatabase()
    db2.load('vectors.pkl')
    db.add_vector("software","I'm working on a software project.")
    db.add_vector("outside","It's very nice outside today.")
    results = db.query('hello world',k=3)
    print("Similar vectors:", results)