# IMPORT IMPORTANT LIBRARIES-
import os
import google.generativeai
from dotenv import load_dotenv
import gradio as gr
import glob
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

model = "gemini-1.5-flash"
db_name = "vector_db"

# USING THE GOOGLE GEMINI API FROM .ENV FILE-
load_dotenv(override=True)
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

folders = glob.glob("knowledge-base/*")
text_loader_kwargs = {"encoding": "utf-8"}

documents = []
for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    folder_docs = loader.load()
    for doc in folder_docs:
        doc.metadata["doc_type"] = doc_type
        documents.append(doc)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

a = len(chunks)
print(a)
doc_types = set(chunk.metadata['doc_type'] for chunk in chunks)
print(f"Document types found: {', '.join(doc_types)}")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    task_type="RETRIEVAL_QUERY"
)

if os.path.exists(db_name):
    # This might fail if the directory is not fully clean or if there's a lock.
    # For robust deletion, consider shutil.rmtree(db_name) if it's safe to delete the whole directory.
    # Otherwise, rely on Chroma's internal collection management.
    try:
        Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()
    except Exception as e:
        print(f"Could not delete existing Chroma collection: {e}. Proceeding to overwrite.")

vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
print(f"Vectorstore created with {vectorstore._collection.count()} documents.")

collection = vectorstore._collection
sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"The vectors have {dimensions:,} dimensions.")

result = collection.get(include=['embeddings', 'documents', 'metadatas'])
vectors = np.array(result['embeddings'])
documents = result['documents']
doc_types_list = [metadata['doc_type'] for metadata in result['metadatas']] # Renamed to avoid conflict with set
# Ensure all doc_types are in the list for consistent coloring
color_map = {
    'products': 'blue',
    'employees': 'green',
    'contracts': 'red',
    'company': 'orange'
}
colors = [color_map.get(t, 'gray') for t in doc_types_list] # Default to gray if unknown doc_type

tsne = TSNE(n_components=2, random_state=42)
reduced_vectors = tsne.fit_transform(vectors)

# Now using the vector embeddings to extract relevant information from the folder's data
MODEL = "gemini-1.5-flash"
llm = ChatGoogleGenerativeAI(temperature=0.7, model=MODEL)
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
retriever = vectorstore.as_retriever()

conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    # REMOVE THIS LINE: question_generator_chain_input_key = 'question',
    output_key='answer'
)

query = "Please explain what Insurellm is in a couple of sentences"
result = conversation_chain.invoke({"question": query, "chat_history": []})
print(result["answer"])