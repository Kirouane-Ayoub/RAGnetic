import cohere
import dspy
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv
from utils import settings

load_dotenv()


class CohereEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, cohere_client):
        self.cohere_client = cohere_client

    def __call__(self, input: Documents) -> Embeddings:
        return self.cohere_client.embed(texts=input).embeddings


# Initialize Cohere client
cohere_client = cohere.Client()

embedding_function = CohereEmbeddingFunction(cohere_client)

cohere_llm = dspy.Cohere(model=settings.COHERE_MODEL)
