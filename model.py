from langchain_community.llms import Ollama
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from bs4 import BeautifulSoup as Soup
from langchain.utils.html import PREFIXES_TO_IGNORE_REGEX, SUFFIXES_TO_IGNORE_REGEX


from config import *
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

global conversation
conversation = None

def init_index():
    if not INIT_INDEX:
        logging.info("continue without initializing index")
        return
    #load the url into text
    documents = RecursiveUrlLoader(
        TARGET_URL,
        max_depth = 4,
        extractor = lambda x: Soup(x, "html.parser").text,
        prevent_outside = True,
        use_async = True,# faster for large tasks
        timeout = 600,
        check_response_status = True,
        #drop trailing / to avoid duplicate pages
        link_regex = (#regex for extracting sub-links from the raw html of a web page
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        )
    ).load()
    logging.info("index creating with %d documents", len(documents))
    
    #split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(documents)
    #create embeddings with huggingface embedding model 'all-MiniLM-L6-v2'
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorbd = Chroma.from_documents(
        documents = documents,
        embeddings = embeddings,
        persist_diretory = INDEX_PERSIST_DICTORY
    )

def init_conversation():
    global conversation

    # load the vector database
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory = INDEX_PERSIST_DICTORY, embedding_function = embeddings)

    # run llama3, expose an api for the llama in 'localhost:11434'
    llm = Ollama(
        model='llama3',
        base_url="http://localhost:11434",
        verbose = True
    ) 

    # create conversation
    conversation = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectordb.as_retriever(),
        return_source_documents = True,
        verbose = True
    )

def chat(question, user_id):
    global conversation

    chat_history = []
    response = conversation({"question": question, "chat_history": chat_history})
    answer = response['answer']

    logging.info("got response from llm - %s", answer)
    #TODO save chat history

    return answer
