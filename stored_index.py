import warnings
warnings.filterwarnings("ignore", message=".*pydantic_v1.*")

from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# print(bool(PINECONE_API_KEY))
# print(PINECONE_API_ENV)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

#Initializing the Pinecone 
# Initialize Pinecone client (new SDK)
pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "medical-bot"

# Create index if it doesn't exist
if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # matches MiniLM model
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_API_ENV)
    )

#Creating Embeddings for Each of the text chunks and storing
# docsearch = PineconeVectorStore.from_existing_index(
#     index_name=INDEX_NAME,
#     embedding=embeddings
# )
docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
docsearch.add_documents(
    text_chunks,
    ids=[f"doc-{i}" for i in range(len(text_chunks))]
)

# res = docsearch.similarity_search("What are allergies?", k=3)
# for i, d in enumerate(res, 1):
#     print(f"\nResult {i}:\n{d.page_content[:300]}...")

