
import os, warnings
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv


warnings.filterwarnings("ignore", message=".*pydantic_v1.*")


from src.helper import download_hugging_face_embeddings
from src.prompt import prompt_template


from pinecone import Pinecone, ServerlessSpec             
from langchain_pinecone import PineconeVectorStore     
from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_REGION  = os.environ.get("PINECONE_API_ENV", "us-east-1")  
INDEX_NAME       = "medical-bot"
DIM              = 384 

# 1) Embeddings
embeddings = download_hugging_face_embeddings()  

# 2) Pinecone client (new SDK)  AND ensure index exists 
pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION),
    )

# 3) Vector store handle (index must be populated already; otherwise use from_texts/add_documents)
docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)

# 4) Prompt + LLM + QA chain
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={"max_new_tokens": 512, "temperature": 0.8}
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT},
)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET","POST"])
def chat():
    msg = request.form["msg"]
    input = msg 
    print(input)
    result = qa({"query" : input})
    print("Response : ", result["result"])
    return str(result["result"])

if __name__ == "__main__":
    app.run(debug=True, port=5001)
