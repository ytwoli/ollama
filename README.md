# local ChatBot
To run the large language model with more privacy, no leaky of personal data, download the llm and run it locally.
## Ollama
a lightweight and flexible framework for local deployment of LLM. Provide a collection of pre-configured models. it quantize the neural network to make it applicable for personal computers
## RAG
incorporates a custom-made dataset, can also dynamically scraped from an online website, user interacte with website's date through API. Document -> scrap -> split -> store in Chorma vector database as vector embeddings
### Data Indexing
Web <- scrape web - RecursiveUrlloader((langchain) -> scraped documents -> RecursiveCharacterText Spliter(Langchain) - split documents -> Chunks -> HuggingfaceEmbedding(langchain) - create vector embedding ->Vector Embedding -> HuggingfaseEmbedding(langchain) - save vector embeddings -> Chroma vector Store
* Scrape Web Data: different document loaders from langchain, e.g. ResursiveUrlLoader is to scrape data from the web as documents
* Split Documents: divide the text into smaller segments, the langchain text splitter divides the text into small semantically meaningful units
* Create Vector Embedding: Convert the texual information into vector embeddings, easy to be grouped, sorted, searched and more
*  Store Vector Embedding in Chroma: ChromaDB is an opensource embedding database, allows for quick retrieval and comparison of text-based data
### Data Querying
![image](./dataquery.png) 
* Semantic Search Vector Database: semantically search through the vector database and find out the most relevant content to the user's query
* Save Query and Response in MongoDB Chat History: to manage conversational memory, the history data is essential in shaping future interactions
