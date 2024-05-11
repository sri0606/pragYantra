import os
import torch
import h5py
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import BertTokenizer, BertModel
from utils.summarize import Summarizer
from . import MODELS_DIR, MEMORY_STREAM_DIR

class VectorDatabase:
    """
    Vector database to aid Memory

    Attributes:
        model: The model used for vectorization.
        vectors (dict): A dictionary to store the vectors.

    Methods:
        __init__(self, model=None): Initializes a VectorDatabase object.
        add_vector(self, key, text): Adds a vector to the database.
        vectorize_text(self, text): Vectorizes the input text.
        query(self, text, k=10): Performs a query on the database.
        save(self, filename): Saves the vectors to a file.
        load(self, filename): Loads the vectors from a file.
    """

    def __init__(self,model=None):
        self.vectors = {}
        self.summarizer = Summarizer()

    def preprocess_text(self, text, threshold_length=100):
        """
        Preprocesses the input text if its too long by summarizing it.

        Args:
            text (str): The input text to preprocess.
            threshold_length (int, optional): The threshold length for summarizing the text. Defaults to 100.
        Returns:
            str: The preprocessed text.
        """
        if len(text)>threshold_length:
            text = self.summarizer.summarize(text)
        return text
    
    def add_vector(self,key,text):
        """
        Adds a vector to the database.

        Args:
            key (str): The key to associate with the vector.
            text (str): The text to vectorize and add to the database.
        """
        preprocess_text = self.preprocess_text(text,threshold_length=100)
        vector_emb = self.vectorize_text(preprocess_text)
        self.vectors[key] = vector_emb

    def vectorize_text(self,text):
        raise NotImplementedError("VectorDatabase.vectorize_text method is not implemented")
    
    def query(self,text,k=10):
        raise NotImplementedError("VectorDatabase.query method is not implemented")
    
    def save(self, filename):
        """
        Saves the vectors to a file.

        Args:
            filename (str): The name of the file to save the vectors to.
        """
        with h5py.File(os.path.join(MEMORY_STREAM_DIR,filename), 'w') as f:
            for key, value in self.vectors.items():
                f.create_dataset(key, data=value)

    def load(self, filename):
        """
        Loads the vectors from a file.

        Args:
            filename (str): The name of the file to load the vectors from.
        """
        with h5py.File(os.path.join(MEMORY_STREAM_DIR,filename), 'r') as f:
            self.vectors = {key: np.array(value) for key, value in f.items()}


class VectorDatabaseSB(VectorDatabase):
    """
    Vector Database using SenctenceBert models
    """
    def __init__(self, model="all-MiniLM-L6-v2"):
       
        try:
            path = os.path.join(MODELS_DIR,model)
            self.model = SentenceTransformer(path)
        except:
            self.model = SentenceTransformer(model)

        super().__init__()

    def vectorize_text(self, text):
        embedding = self.model.encode(text, convert_to_tensor=True)
        return embedding.numpy()

    def query(self, text, k=5):
        query_embedding = self.model.encode(text, convert_to_tensor=True)
        vectors = torch.stack([torch.from_numpy(v) for v in self.vectors.values()])
        hits = util.semantic_search(query_embedding, vectors, top_k=k)  # Unsqueeze the query_embedding to match the dimensions
        return [(list(self.vectors.keys())[hit['corpus_id']], hit['score']) for hit in hits[0]]

class VectorDatabaseBert(VectorDatabase):
    """
    A class representing a vector database using BERT model.

    Attributes:
        model (BertModel): The BERT model used for vectorization.
        tokenizer (BertTokenizer): The BERT tokenizer used for tokenization.
        vectors (dict): A dictionary to store the vectors.
    """

    def __init__(self, model="bert-base-uncased"):
        """
        Initializes a VectorDatabase object.

        Args:
            model (str, optional): The name of the model to use for vectorization.
        """
        try:
            bert_path = os.path.join(MODELS_DIR,model)
            self.model = BertModel.from_pretrained(bert_path).eval()
            self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        except:
            self.model = BertModel.from_pretrained(model).eval()
            self.tokenizer = BertTokenizer.from_pretrained(model)

        super().__init__()

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
    


import time
if __name__ == '__main__':
    db = VectorDatabaseBert()
    db1 = VectorDatabaseSB()
    start_time=time.time()
    db.add_vector('hello', 'Hello, world!')
    db.add_vector('goodbye', 'Goodbye, world!')
    print('Adding time b:',time.time()-start_time)
    
    start_time=time.time()  
    db1.add_vector('hello', 'Hello, world!')
    db1.add_vector('goodbye', 'Goodbye, world!')
    print('Adding time sb:',time.time()-start_time)
    db.save('vectors.h5')
    db1.save('vectorssb.h5')

    db = VectorDatabaseBert()
    db1 = VectorDatabaseSB()
    db.load('vectors.h5')
    db1.load('vectorssb.h5')
    db.add_vector("software","I'm working on a software project.")
    db1.add_vector("software","I'm working on a software project.")

    start_time=time.time()
    results = db.query('hello world',k=3)
    print("Similar vectors b:", results)
    print('Query time b:',time.time()-start_time)

    start_time=time.time()
    results1 = db1.query('hello world',k=3)
    print("Similar vectors sb:", results1)
    print('Query time sb:',time.time()-start_time)