import os
import torch
import h5py
from datetime import datetime
import base64
import pickle
import numpy as np
from typing import Dict, List
from sentence_transformers import SentenceTransformer, util
from transformers import BertTokenizer, BertModel
from .utils.summarize import Summarizer
from . import MODELS_DIR, MEMORY_STREAM_DIR

class MemoryNode:
    """
    Representation of a memory node.
    """
    def __init__(self, timestamp:str | int, text_embedding: np.ndarray, summary:str=None):
        self.timestamp = timestamp
        self.summary = summary
        self.embedding = text_embedding
    
    def __str__(self):
        return f"MemoryNode: {self.format_timestamp(self.timestamp)} - {self.summary}"
    
    def __repr__(self):
        return f"MemoryNode('{self.timestamp}','{self.embedding}', '{self.summary}')"
    
    @staticmethod
    def format_timestamp(timestamp):
        return f"{timestamp[6:8]}/{timestamp[4:6]}/{timestamp[:4]} {timestamp[8:10]}:{timestamp[10:12]}"
    
    def to_dict(self):
        """
        Returns a dictionary representation of the MemoryNode object.

        Returns:
            dict: A dictionary with keys 'timestamp', 'summary', and 'embedding'.
        """
        return {
            'timestamp': self.timestamp,
            'summary': self.summary,
            'embedding': self.embedding.tolist()  # Convert numpy array to list
        }

    @classmethod
    def from_dict(cls, data):
        """
        Creates a MemoryNode object from a dictionary.

        Args:
            data (dict): A dictionary with keys 'timestamp', 'summary', and 'embedding'.

        Returns:
            MemoryNode: A new MemoryNode object.
        """
        return cls(data['timestamp'], np.array(data['embedding']), data['summary'])
    
class MemoryStoreBase:
    """
    Vector database base class to aid Memory

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

    def __init__(self):
        # Memory nodes is a dictionary with timestamp as key and MemoryNode as value
        self.memory_nodes:Dict[str, MemoryNode] = {}
        self.summarizer = Summarizer()

    def __str__(self):
        return f"Memory with {len(self.memory_nodes)} nodes"

    def __add__(self, other):
        self.integrate_databases(other)
        return self
    
    def __len__(self):
        return len(self.memory_nodes)
    

    def __getitem__(self, key) -> MemoryNode | List[MemoryNode]:
        if isinstance(key, datetime):
            key = key.strftime("%Y%m%d%H%M")
        elif isinstance(key, int):
            key = str(key)

        if len(key) == 12:  # Full format
            return self.memory_nodes[key]
        else:
            summaries = []
            for timestamp_key, memory_node in self.memory_nodes.items():
                if timestamp_key.startswith(key):
                    summaries.append((memory_node))
            return summaries

    def preprocess_text(self, text, threshold_length=200):
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
    
    def pre_seed_memory(self, data: dict, save_path=None):
        """
        Pre-seed the memory with given data kind of like initial memory configuration. 

        Args:
            data (dict): A dictionary of key-value pairs to seed the memory with. The key is the timestamp in the format YYMMDDHHMM (with minutes set to either 00 or 30) and the value is a tuple of the summary and text.
            save_path (str, optional): The path to save the memory to. Defaults to None.
        """
        if not data or not isinstance(data, dict):
            raise ValueError("Data must be a non-empty dictionary.")

        for timestamp,(summary,text)  in data.items():
            self.add_vector(timestamp_key=timestamp, text=text,summary=summary )
        if save_path:
            self.save(save_path)

    def add_vector(self, timestamp_key, text, summary=None):
        """
        Adds a vector to the database.

        Args:
            timestamp_key (str): The timestamp key to associate with the vector. (timestamp YYmmddHHMM (minutes can either be 00 or 30 as memory is added every 30mins))
            text (str): The text to vectorize and add to the database.
            summary (list, optional): The summary associated with the vector. Defaults to None.
        """
        preprocess_text = self.preprocess_text(text, threshold_length=100)
        vector_emb = self.vectorize_text(preprocess_text)
        self.memory_nodes[timestamp_key] = MemoryNode(timestamp=timestamp_key, summary=summary, text_embedding=vector_emb)

    def vectorize_text(self,text):
        raise NotImplementedError("VectorDatabase.vectorize_text method is not implemented")
    
    def query(self, text, date=None, date_type='day', k=5, days_threshold=2):
        raise NotImplementedError("VectorDatabase.query method is not implemented")
    
    def save(self, filename):
        """
        Saves the vectors to a file.

        Args:
            filename (str): The name of the file to save the vectors to.
        """
        with h5py.File(os.path.join(MEMORY_STREAM_DIR,filename), 'w') as f:
            for key, memory_node in self.memory_nodes.items():
                # Serialize the value to a byte string and encode it to base64 (to avoid NULL errors) )
                serialized_value = base64.b64encode(pickle.dumps(memory_node.to_dict()))
                f.create_dataset(key, data=serialized_value)

    def load(self, filename):
        """
        Loads the vectors from a file.

        Args:
            filename (str): The name of the file to load the vectors from.
        """
        with h5py.File(os.path.join(MEMORY_STREAM_DIR,filename), 'r') as f:
            for key in f.keys():
                # Decode the value from base64 and deserialize it from a byte string
                data = pickle.loads(base64.b64decode(f[key][()]))
                self.memory_nodes[key] = MemoryNode.from_dict(data)

    def integrate_databases(self, source_db):
        """
        Integrates the data from the source database into the target database.

        Args:
            source_db (VectorDatabase): The source database.
        """
        if not isinstance(source_db, type(self)):
            raise TypeError("Source database must be of the same type as the target database")

        if self is source_db:
            raise ValueError("Cannot integrate a database with itself")

        for key, node in source_db.memory_nodes.items():
            self.memory_nodes[key] = node

class MemoryStoreSB(MemoryStoreBase):
    """
    Memory store  using SenctenceBert models
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

    def query(self, text, date=None, date_type='day', k=5, days_threshold=2):
        """
        Query embeddings for similar vectors. 
        You can query a text with specific day, month or year. The 'date' parameter should be in the format 'YYYYMMDD' for day, 'YYYYMM' for month, and 'YYYY' for year.

        Args:
            text (str): The query text.
            date (str, optional): Specific date to query for. Defaults to None.
            date_type (str, optional): The type of date query ('day', 'month', or 'year'). Defaults to 'day'.
            k (int, optional): The number of most similar vectors to return. Defaults to 5.
            days_threshold (int, optional): The threshold for the date. Defaults to 2.
        """
        
        if date is not None:
            nearest_nodes = {timestamp_key: node for timestamp_key, node in self.memory_nodes.items() 
                        if date is None or 
                        (date_type == 'day' and abs((datetime.strptime(timestamp_key, "%Y%m%d%H%M") - datetime.strptime(date, "%Y%m%d")).days) <= days_threshold) or 
                        (date_type == 'month' and datetime.strptime(timestamp_key, "%Y%m%d%H%M").month == datetime.strptime(date, "%Y%m").month) or 
                        (date_type == 'year' and datetime.strptime(timestamp_key, "%Y%m%d%H%M").year == int(date))}
        
        elif date is None or len(nearest_nodes)==0:
            print("No vectors found for the given date.\n Searching entire database....")
            nearest_nodes = self.memory_nodes

        query_embedding = self.model.encode(text, convert_to_tensor=True)
        vectors = torch.stack([torch.from_numpy(node.embedding) for node in nearest_nodes.values()])
        vectors = vectors.type_as(query_embedding)  # Ensure vectors is the same type as query_embedding
        hits = util.semantic_search(query_embedding, vectors, top_k=k)
        return [(key, hit['score'], nearest_nodes[key].summary) for hit in hits[0] for key in [list(nearest_nodes.keys())[hit['corpus_id']]]]
    
class MemoryStoreBert(MemoryStoreBase):
    """
    DEPRECATED: A class representing a vector database using BERT model.

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
        similarities = [(key, self.cosine_similarity(query_embedding.numpy(), node.embedding)) for key, node in self.memory_nodes.items()]
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
    db1 = MemoryStoreSB()
    
    #genereated with chatgpt
    random_initial_memory = {
        '202303261530': ('Family Vacation to Italy',
        'Exploring ancient ruins and indulging in delicious Italian cuisine, soaking in the rich history and vibrant culture of Italy.'),
        '202210252200': ('Hiking Mount Kilimanjaro',
        'Reaching the summit of Mount Kilimanjaro after days of challenging trekking, feeling a sense of accomplishment and awe at the breathtaking views.'),
        '202009210500': ('First Day of College',
        'Navigating the bustling campus with a mix of excitement and nervousness, eager to embark on this new academic journey.'),
        '202107141830': ('Volunteering at a Homeless Shelter',
        'Spending a day serving meals and interacting with residents at a homeless shelter, feeling humbled and grateful for the opportunity to make a difference.'),
        '192109081830': ('Road Trip Across America',
        'Embarking on a cross-country road trip with friends, discovering hidden gems and creating unforgettable memories along the way.'),
        '192305222200': ('High School Prom',
        'Dancing the night away with friends, feeling like royalty in my elegant gown and making memories that would last a lifetime.'),
        '182011261100': ('Attending a Music Festival',
        'Immersing myself in the electrifying atmosphere of a music festival, dancing to my favorite bands and connecting with fellow music lovers.'),
        '202209051830': ('Skydiving for the First Time',
        'Feeling an exhilarating rush of adrenaline as I leaped from the plane and soared through the sky, overcome with a sense of freedom and thrill.'),
        '202003050730': ('Watching the Northern Lights',
        'Gazing up at the mesmerizing display of colors dancing across the night sky, feeling humbled by the beauty and wonder of the natural world.'),
        '192008230430': ('Cooking a Homemade Meal with Family',
        'Gathering in the kitchen with loved ones, sharing laughter and stories as we prepared a delicious homemade meal together.'),
        '202002252230': ("Celebrating New Year's Eve in Times Square",
        'Counting down to midnight amidst the sea of revelers in Times Square, feeling a sense of unity and excitement as we welcomed the new year.'),
        '202005082200': ("Attending a Friend's Wedding",
        'Witnessing the love and joy between two dear friends as they exchanged vows, feeling honored to be part of their special day.'),
        '192101082100': ('Scuba Diving in the Great Barrier Reef',
        'Exploring the vibrant underwater world of the Great Barrier Reef, swimming alongside colorful fish and majestic coral formations.'),
        '202206102000': ('Completing a Marathon',
        'Crossing the finish line of a marathon after months of training and dedication, feeling a surge of pride and accomplishment.'),
        '202211161630': ('Planting a Garden',
        'Getting my hands dirty in the soil and watching with joy as my garden bloomed with colorful flowers and delicious fruits and vegetables.')
    }


    # print("Initial memory config")
    # start_time = time.time()
    # db1.pre_seed_memory(data = random_initial_memory, save_path='memory.h5')
    # print('Adding time sb:', time.time() - start_time)

    # Load the vectors from files (if needed)
    db1.load('memory.h5')

    # Example queries
    query_texts = [
        "Recall the experience of hiking Mount Kilimanjaro.",
        "Retrieve memory of scuba diving in the Great Barrier Reef.",
        "Music and songs",
        "Covid 19 pandemic",
    ]
    dates = [None,None,1820,2021]
    for i,query in enumerate(query_texts):

        start_time = time.time()
        results1 = db1.query(query,date=dates[i],date_type='year', k=3)
        print("Similar vectors sb:", results1)
        print('Query time sb:', time.time() - start_time)