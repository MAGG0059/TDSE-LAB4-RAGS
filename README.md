
# TDSE LAB4 RAG MODEL Manuel Alejandro Guarnizo

I built a Retrieval-Augmented Generation (RAG) system using LangChain and Google Gemini that answers questions about LLM-powered autonomous agents based on a blog post by Lilian Weng. The system follows a two-stage pipeline: indexing and retrieval-generation. For indexing, I used WebBaseLoader to load the blog post, RecursiveCharacterTextSplitter to divide it into 66 chunks of 1000 characters each with 200-character overlap, and GoogleGenerativeAIEmbeddings with the gemini-embedding-001 model to convert these chunks into vector representations stored in an InMemoryVectorStore. For retrieval and generation, I implemented two approaches: a RAG agent with a custom retrieve_context tool that performs similarity searches and returns both serialized context and raw documents, and a simplified RAG chain using dynamic prompts that always injects retrieved context before each query. The agent successfully handles complex queries requiring multiple retrieval steps, such as asking about task decomposition methods and their extensions, demonstrating how it can iteratively search for information. The system showcases key RAG concepts including semantic search, context injection, tool-based retrieval, and the trade-offs between agentic and chain-based approaches.


## In this repository are the notebooks:
Rag-Model.ipynb



## Setup Instructions (dependencies)
python-dotenv==1.0.0
google-generativeai>=0.8.0
langchain>=0.3.0
langchain-google-genai>=2.0.0
langchain-community>=0.3.0
langchain-text-splitters>=0.3.0
langgraph>=0.3.0
beautifulsoup4>=4.12.0
bs4>=0.0.2
sentence-transformers>=2.2.0

API Key Setup (IMPORTANT)

-Create a .env file in the project root (this file is listed in .gitignore and will NOT be pushed to GitHub)

-Add your Google API key to the .env file:

GOOGLE_API_KEY=your-actual-api-key-here
Get your free API key from: https://aistudio.google.com

### Components
Document Loader	WebBaseLoader - Carga contenido web desde URL con BeautifulSoup
Text Splitter	RecursiveCharacterTextSplitter - Divide documentos en chunks de 1000 caracteres con overlap de 200
Embeddings Model	GoogleGenerativeAIEmbeddings - Convierte texto a vectores (modelo: gemini-embedding-001)
Vector Store	InMemoryVectorStore - Almacena y busca vectores por similitud
Chat Model	ChatGoogleGenerativeAI / init_chat_model - Modelo Gemini para responder preguntas
Retrieval Tool	retrieve_context - Herramienta que busca documentos relevantes en el vector store
Agent	create_agent - Agente que decide cuándo usar la herramienta de retrieval

### Arquitecture LangChainLLLM

Web Source → Document Loader → Text Splitter → Embeddings → Vector Store ← Retrieval Tool ← Agent ← User Query → Context + Query → LLM → Response

### Requirements
- **Python 3.10+** (compatible with most Python 3.x versions)
- **Jupyter Notebook** or **JupyterLab**
- Python libraries: `numpy`, `matplotlib`, `pandas`

### Installation Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>


