import chromadb
from dspy.retrieve.chromadb_rm import ChromadbRM
from utils import settings
from utils.models import embedding_function
from utils.pdf_to_text import extract_texts_from_folder
from utils.spliter import regex_splitter

client = chromadb.PersistentClient(path=settings.CHROMA_DATA_PATH)

print("------ Create or Get ChromaDB Collection  ------ ")

collection = client.get_or_create_collection(
    name=settings.COLLECTION_NAME,
    embedding_function=embedding_function,
    metadata={"hnsw:space": "cosine"},
)


print("------ Text Extraction  ------ ")

TEXT = extract_texts_from_folder(settings.KNOWLEDGE_BASE_PATH)

print("------ Text Split ------ ")

documents = regex_splitter(
    text=TEXT, chunk_size=settings.CHUNK_SIZE, overlap=settings.OVERLAP
)

print("------ Text Embedding and Storing ------ ")

collection.add(
    documents=documents,
    ids=[f"id{i}" for i in range(len(documents))],
)
print("------ Create DSPY Retriever Model ------ ")
retriever_model = ChromadbRM(
    settings.COLLECTION_NAME,
    settings.CHROMA_DATA_PATH,
    embedding_function=embedding_function,
    k=settings.THE_NUMBER_OF_DOCUMENTS_TO_FETCH,
)
print("------ Done !! ------ ")
