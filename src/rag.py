# rag.py
import chromadb
from sentence_transformers import SentenceTransformer
import os
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define a wrapper class for the Sentence Transformer embedding function to satisfy ChromaDB's requirements
class SentenceTransformerChromaEmbeddingFunction:
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        """
        Initializes the SentenceTransformer model.
        Downloads the model if not already cached.
        """
        try:
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
            self._model_name = model_name
            print(f"SentenceTransformer model '{model_name}' loaded.")
        except Exception as e:
            print(f"Error loading SentenceTransformer model '{model_name}': {e}")
            raise

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of texts using the SentenceTransformer model.
        The parameter name 'input' is required by ChromaDB's EmbeddingFunction interface.
        """
        embeddings = self.model.encode(input).tolist()
        return embeddings

    def name(self) -> str:
        """
        Returns the name of the embedding function, as expected by ChromaDB.
        """
        return self._model_name

class DocumentRag:
    def __init__(self, collection_name: str = "document_chunks"):
        self.client = chromadb.Client()
        self.embedding_function_instance = SentenceTransformerChromaEmbeddingFunction()
        print("Sentence Transformer Embedding Function Wrapper initialized.")

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function_instance
        )
        print(f"ChromaDB collection '{collection_name}' initialized with Sentence Transformer embeddings.")

    def add_documents(self, texts: list[str], metadatas: list[dict] = None) -> None:
        """
        Adds text chunks and their embeddings to the ChromaDB collection.

        Args:
            texts (list[str]): A list of text chunks to add.
            metadatas (list[dict], optional): A list of metadata dictionaries,
                                               one for each text chunk.
                                               Defaults to None.
        """
        if not texts:
            print("No texts provided to add.")
            return

        ids = [f"doc_{i}" for i in range(len(texts))]

        # Ensure metadatas is a list of dictionaries with the same length as texts
        # and that each dictionary is not empty.
        processed_metadatas = []
        if metadatas is None:
            # Create default metadata for each chunk
            processed_metadatas = [{"source": "uploaded_document"} for _ in range(len(texts))]
        elif len(metadatas) != len(texts):
            print(f"Warning: Length of provided metadatas ({len(metadatas)}) does not match length of texts ({len(texts)}). Defaulting to basic metadata.")
            processed_metadatas = [{"source": "uploaded_document"} for _ in range(len(texts))]
        else:
            # Use provided metadatas, but ensure each is a non-empty dict
            for i, meta in enumerate(metadatas):
                if not isinstance(meta, dict) or not meta: # Check if not a dict or if empty
                    processed_metadatas.append({"source": "uploaded_document", "original_index": i})
                else:
                    processed_metadatas.append(meta)

        try:
            self.collection.add(
                documents=texts,
                metadatas=processed_metadatas,
                ids=ids
            )
            print(f"Added {len(texts)} documents to ChromaDB.")
        except Exception as e:
            print(f"Error adding documents to ChromaDB: {e}")

    def query_documents(self, query_text: str, n_results: int = 5) -> list[str]:
        """
        Queries the ChromaDB collection for relevant text chunks.

        Args:
            query_text (str): The query string.
            n_results (int): The number of top relevant results to retrieve.

        Returns:
            list[str]: A list of relevant text chunks.
        """
        if not query_text:
            return []

        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            if results and results['documents']:
                return results['documents'][0]
            return []
        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
            return []

    def clear_collection(self):
        """
        Clears all documents from the current collection.
        Useful for resetting the database for new document uploads.
        """
        try:
            self.client.delete_collection(name=self.collection.name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection.name,
                embedding_function=self.embedding_function_instance
            )
            print(f"ChromaDB collection '{self.collection.name}' cleared and re-initialized.")
        except Exception as e:
            print(f"Error clearing ChromaDB collection: {e}")

if __name__ == '__main__':
    rag = DocumentRag(collection_name="test_document_chunks_st")
    rag.clear_collection()
    dummy_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is a rapidly developing field.",
        "Machine learning is a subset of AI that focuses on algorithms.",
        "Natural language processing deals with the interaction between computers and human language.",
        "Vector databases are optimized for storing and querying high-dimensional vectors."
    ]
    rag.add_documents(dummy_texts)
    query = "What is AI and machine learning?"
    print(f"\nQuerying for: '{query}'")
    relevant_chunks = rag.query_documents(query, n_results=2)
    print("Relevant chunks found:")
    for chunk in relevant_chunks:
        print(f"- {chunk}")
    query_2 = "What are vector databases used for?"
    print(f"\nQuerying for: '{query_2}'")
    relevant_chunks_2 = rag.query_documents(query_2, n_results=1)
    print("Relevant chunks found:")
    for chunk in relevant_chunks_2:
        print(f"- {chunk}")
    print("\nQuerying with empty string:")
    empty_query_results = rag.query_documents("")
    print(f"Results for empty query: {empty_query_results}")
    rag.clear_collection()
    print("\nAfter clearing, querying again:")
    cleared_results = rag.query_documents(query)
    print(f"Results after clear: {cleared_results}")
