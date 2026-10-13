# document loader 
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings



# Extract Data From the PDF File
def load_pdf(path):
    'load the pdf'
    loader=DirectoryLoader(
        path,
        glob='*.pdf',
        loader_cls=PyPDFLoader
    )
    documents=loader.load()
    return documents

def filter_to_minimal_docs(docs:list[Document])->list[Document]:
    minimal_docs:List[Document]=[]
    for doc in docs:
        src=doc.metadata['source']
        minimal_docs.append(Document(page_content=doc.page_content,metadata={'source':src}))
    return minimal_docs


 # Split the Data into Text Chunks


def text_split(minimal_doc):

    "function for chunking the minial docs"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=20)
    text_chunks=text_splitter.split_documents(minimal_doc)
    return text_chunks



# Download the Embeddings From HuggingFace
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings