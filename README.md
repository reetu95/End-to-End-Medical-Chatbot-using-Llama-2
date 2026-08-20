# End-to-End Medical Chatbot Using Llama 2

A retrieval-augmented generation (RAG) chatbot that answers medical questions using information retrieved from a medical PDF knowledge base. The application combines a locally running Llama 2 model with LangChain, Hugging Face sentence embeddings, Pinecone, and a Flask web interface.

## Features

- Answers questions using context retrieved from medical reference material
- Loads and processes PDF documents automatically
- Creates semantic embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- Stores and searches document vectors with Pinecone
- Runs a quantized Llama 2 7B Chat model locally through CTransformers
- Provides a simple Flask, Bootstrap, and jQuery chat interface
- Uses a grounded prompt that tells the model to admit when the answer is unknown

## How It Works

The project has two main workflows.

### 1. Knowledge-base indexing

```text
PDF documents
    -> PyMuPDF document loader
    -> 500-character chunks with 20-character overlap
    -> MiniLM embeddings (384 dimensions)
    -> Pinecone index named "medical-bot"
```

Run `stored_index.py` to build this index before starting the chatbot.

### 2. Question answering

```text
User question
    -> Question embedding
    -> Retrieve the two most relevant Pinecone records
    -> Add the retrieved text to the prompt
    -> Generate an answer with local Llama 2
    -> Display the answer in the Flask chat interface
```

This technique is called **retrieval-augmented generation**, or **RAG**. The Llama 2 model is not trained or fine-tuned by this project. Instead, relevant information is retrieved and supplied to the model at question time.

## Technology Stack

- Python
- Flask
- LangChain
- Meta Llama 2 7B Chat
- CTransformers
- Hugging Face Sentence Transformers
- Pinecone
- PyMuPDF
- Bootstrap and jQuery

## Project Structure

```text
.
├── app.py                    # Flask server and RAG question-answering chain
├── stored_index.py           # Builds and populates the Pinecone index
├── requirements.txt          # Python dependencies
├── setup.py                  # Local package configuration
├── data/
│   └── Medical_book.pdf      # Medical reference knowledge base
├── model/
│   └── instruction.txt       # Model download instructions
├── research/
│   └── trails.ipynb          # Development and experimentation notebook
├── src/
│   ├── helper.py             # PDF loading, splitting, and embedding helpers
│   └── prompt.py             # Retrieval QA prompt template
├── static/
│   └── style.css             # Chat interface styling
└── templates/
    └── chat.html             # Chat interface markup and JavaScript
```

## Prerequisites

Before starting, install or create accounts for the following:

- Python 3.10 recommended
- A [Pinecone](https://www.pinecone.io/) account and API key
- At least 8 GB of available RAM; more may improve local model performance
- Approximately 4 GB of disk space for the quantized model, in addition to the project dependencies

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/reetu95/End-to-End-Medical-Chatbot-using-Llama-2.git
cd End-to-End-Medical-Chatbot-using-Llama-2
```

### 2. Create a virtual environment

Using `venv`:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\activate
```

Alternatively, using Conda:

```bash
conda create --name mchatbot python=3.10 -y
conda activate mchatbot
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Pinecone

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_API_ENV=us-east-1
```

`PINECONE_API_ENV` is used as the AWS region for the serverless Pinecone index. Change it if your Pinecone project uses another supported region.

Never commit the `.env` file or expose your API key publicly.

### 5. Download the Llama 2 model

Download `llama-2-7b-chat.ggmlv3.q4_0.bin` from the [Llama 2 7B Chat GGML repository on Hugging Face](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main).

Place the file at:

```text
model/llama-2-7b-chat.ggmlv3.q4_0.bin
```

The expected project layout is:

```text
model/
├── instruction.txt
└── llama-2-7b-chat.ggmlv3.q4_0.bin
```

The model file is approximately 3.8 GB and is intentionally excluded from Git.

Review and comply with the model's license and acceptable-use policy before downloading or using it.

## Build the Pinecone Index

The repository includes a medical reference PDF in `data/`. To process every PDF in that directory and upload its chunks to Pinecone, run:

```bash
python stored_index.py
```

This operation may take several minutes. It downloads the MiniLM embedding model on its first run, processes the PDFs, and sends document embeddings and text metadata to Pinecone.

You normally need to run this command only when:

- setting up the project for the first time;
- changing the documents in `data/`; or
- rebuilding the Pinecone index.

## Run the Application

Start the Flask server:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5001
```

Enter a medical question in the chat box. The first response can take longer because the embedding and Llama 2 models must be loaded into memory.

## Configuration

The principal settings are currently defined in the source code:

| Setting | Default | Location |
|---|---:|---|
| Pinecone index | `medical-bot` | `app.py`, `stored_index.py` |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` | `src/helper.py` |
| Embedding dimensions | `384` | `app.py`, `stored_index.py` |
| Chunk size | `500` characters | `src/helper.py` |
| Chunk overlap | `20` characters | `src/helper.py` |
| Retrieved chunks | `2` | `app.py` |
| Maximum generated tokens | `512` | `app.py` |
| Temperature | `0.8` | `app.py` |
| Flask port | `5001` | `app.py` |

If you change the embedding model, make sure the Pinecone index dimension matches the new model's output dimension. An existing index with a different dimension must be recreated.

## API Routes

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/` | Displays the chatbot interface |
| `POST` | `/get` | Accepts the form field `msg` and returns a generated answer |

Example request:

```bash
curl -X POST http://127.0.0.1:5001/get \
  --data-urlencode "msg=What are allergies?"
```

## Using Your Own Documents

1. Add one or more PDF files to `data/`.
2. Confirm that you have permission to process and store their contents.
3. Run `python stored_index.py` again.
4. Restart the Flask application.

For a clean replacement of the knowledge base, remove or recreate the existing `medical-bot` index in Pinecone before indexing the new documents. Otherwise, records left over from an older, larger dataset may remain in the index.

Do not upload confidential patient information or protected health information unless your complete environment, data handling, provider agreements, access controls, and compliance obligations have been independently reviewed.

## Known Limitations

- The bundled knowledge base is limited and may contain outdated information.
- Retrieval uses only the two nearest chunks and may omit relevant context.
- Generated answers can still be incomplete, incorrect, or hallucinated.
- The web interface does not maintain conversational memory between questions.
- Retrieved source documents are not currently displayed as citations.
- The application does not implement medical emergency detection or clinical safety checks.
- Authentication, rate limiting, production logging, and automated tests are not included.
- Flask debug mode is enabled in the current development configuration and must be disabled before deployment.
- The current frontend should be hardened against untrusted HTML before public deployment.
- The pinned packages use a legacy GGML/CTransformers stack and may require dependency adjustments on newer systems.

## Troubleshooting

### Model file not found

Confirm that the filename and location exactly match:

```text
model/llama-2-7b-chat.ggmlv3.q4_0.bin
```

### Pinecone authentication or region error

Confirm that `.env` exists in the project root and contains valid values:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_API_ENV=us-east-1
```

Also confirm that the chosen Pinecone region is available for your account.

### Pinecone dimension mismatch

The `all-MiniLM-L6-v2` embedding model produces 384-dimensional vectors. Recreate the Pinecone index with dimension `384` if it was originally created with another dimension.

### Empty or irrelevant answers

Make sure `python stored_index.py` completed successfully and that the `medical-bot` index contains records. Also confirm that the question is covered by the PDFs in `data/`.

### Dependency resolution error involving `tokenizers`

The current `transformers==4.36.0` release expects a `tokenizers` version below `0.19`. If pip reports a conflict, change the tokenizers requirement to:

```text
tokenizers>=0.14,<0.19
```

Then reinstall the dependencies in a clean virtual environment.

## Production Considerations

Before exposing this application to users, consider:

- replacing Flask's development server with a production WSGI server;
- disabling debug mode;
- escaping or sanitizing all rendered messages;
- adding authentication, authorization, and rate limiting;
- displaying the retrieved sources with every answer;
- adding medical disclaimers and emergency escalation paths in the interface;
- implementing automated retrieval and answer-quality evaluations;
- adding monitoring without logging sensitive user information; and
- obtaining clinical, legal, privacy, and security review.

## Contributing

Contributions are welcome. To propose a change:

1. Fork the repository.
2. Create a feature branch.
3. Make and test your changes.
4. Commit the changes with a clear message.
5. Open a pull request describing the motivation and implementation.

## License

The source code is available under the [MIT License](LICENSE).

The Llama 2 model and the documents used as a knowledge base may have separate licenses and usage restrictions. Those assets are not automatically covered by the repository's MIT License.

## Author

**Reetu Thimmaiah**

- GitHub: [@reetu95](https://github.com/reetu95)
