# local ChatBot
To run the large language model with more privacy, no leaky of personal data, download the llm and run it locally.
## Ollama
a lightweight and flexible framework for local deployment of LLM. Provide a collection of pre-configured models. it quantize the neural network to make it applicable for personal computers
## RAG
incorporates a custom-made dataset, can also dynamically scraped from an online website, user interacte with website's date through API. Document -> scrap -> split -> store in Chorma vector database as vector embeddings
### Data Indexing
Web <- scrape web - RecursiveUrlloader((langchain) -> scraped documents -> RecursiveCharacterText Spliter(Langchain) - split documents -> Chunks -> HuggingfaceEmbedding(langchain) - create vector embedding ->Vector Embedding -> HuggingfaseEmbedding(langchain) - save vector embeddings -> Chroma vector Store

### Data Querying
