# rag.py
import chromadb
from sentence_transformers import SentenceTransformer
import os
from typing import List, Dict
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
    def __init__(self, collection_name: str = "document_chunks", persist_directory: str = "./chroma_db"):
        """
        Initializes the ChromaDB client and collection.
        Uses PersistentClient for local, file-based storage.
        """
        # Initialize PersistentClient with a specified directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_function = SentenceTransformerChromaEmbeddingFunction()
        self.collection_name = collection_name
        
        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function # Pass the instance here
        )
        print(f"ChromaDB collection '{self.collection_name}' initialized with Sentence Transformer embeddings.")

    def add_documents(self, texts: List[str], metadatas: List[Dict] = None) -> None:
        """
        Adds text documents to the ChromaDB collection.

        Args:
            texts (List[str]): A list of text strings to add.
            metadatas (List[Dict], optional): A list of metadata dictionaries,
                                               one for each text. Defaults to None.
        """
        if not texts:
            print("No texts provided to add to ChromaDB.")
            return

        # ChromaDB requires unique IDs for each document
        ids = [f"chunk_{i}" for i in range(len(texts))]

        # Ensure metadatas list matches the texts list length
        if metadatas is None:
            metadatas = [{"source": "uploaded_document"} for _ in texts]
        elif len(metadatas) != len(texts):
            raise ValueError("Length of metadatas must match length of texts.")

        try:
            self.collection.add(
                documents=texts,
                metadatas=metadatas, # metadatas should be a list of dicts
                ids=ids
            )
            print(f"Added {len(texts)} documents to ChromaDB.")
        except Exception as e:
            print(f"Error adding documents to ChromaDB: {e}")

    def query_documents(self, query_text: str, n_results: int = 1) -> List[str]:
        """
        Queries the ChromaDB collection for relevant documents.

        Args:
            query_text (str): The query string.
            n_results (int): The number of most relevant results to return.

        Returns:
            List[str]: A list of relevant document content strings.
        """
        if not query_text:
            print("Query text is empty. Returning no results.")
            return []
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            # results['documents'][0] contains the list of document contents
            # Ensure it handles cases where no documents are returned
            if results and results['documents'] and len(results['documents']) > 0:
                return results['documents'][0]
            else:
                return []
        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
            return []

    def clear_collection(self):
        """
        Deletes the entire collection. Useful for resetting the database.
        Note: For PersistentClient, this removes the collection from the client's view.
        The underlying files will remain until the persist_directory is manually cleared.
        """
        try:
            # Delete the collection
            self.client.delete_collection(name=self.collection_name)
            # Re-create the collection to ensure it's empty and ready for new data
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            print(f"ChromaDB collection '{self.collection_name}' cleared and re-initialized.")
        except Exception as e:
            print(f"Error clearing ChromaDB collection: {e}")

if __name__ == '__main__':
    # When running this directly, it will create a 'test_chroma_db' directory
    rag = DocumentRag(collection_name="test_document_chunks_st", persist_directory="./test_chroma_db")
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
    query_2 = "What are vector databases used for??"
    print(f"\nQuerying for: '{query_2}'")
    relevant_chunks_2 = rag.query_documents(query_2, n_results=1)
    print("Relevant chunks found:")
    for chunk in relevant_chunks_2:
        print(f"- {chunk}")
    print("\nQuerying with empty string:")
    empty_query_results = rag.query_documents("")
    print(f"Results for empty query: {empty_query_results}")
    rag.clear_collection()
    # Clean up the test directory after execution
    import shutil
    if os.path.exists("./test_chroma_db"):
        shutil.rmtree("./test_chroma_db")

