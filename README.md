# local ChatBot
based on [link](https://medium.com/rahasak/session-based-chatbot-for-website-content-utilizing-openai-gpt-4-llm-langchain-ef09e0706767)
To run the large language model with more privacy, no leaky of personal data, download the llm and run it locally.
## Ollama
a lightweight and flexible framework for local deployment of LLM. Provide a collection of pre-configured models. it quantize the neural network to make it applicable for personal computers
## RAG
incorporates a custom-made dataset, can also dynamically scraped from an online website, user interacte with website's date through API. Document -> scrap -> split -> store in Chorma vector database as vector embeddings
### Web-Scrabble
#### Data Indexing
Web <- scrape web - RecursiveUrlloader((langchain) -> scraped documents -> RecursiveCharacterText Spliter(Langchain) - split documents -> Chunks -> HuggingfaceEmbedding(langchain) - create vector embedding ->Vector Embedding -> HuggingfaseEmbedding(langchain) - save vector embeddings -> Chroma vector Store
* Scrape Web Data: different document loaders from langchain, e.g. ResursiveUrlLoader is to scrape data from the web as documents
* Split Documents: divide the text into smaller segments, the langchain text splitter divides the text into small semantically meaningful units
* Create Vector Embedding: Convert the texual information into vector embeddings, easy to be grouped, sorted, searched and more
*  Store Vector Embedding in Chroma: ChromaDB is an opensource embedding database, allows for quick retrieval and comparison of text-based data
#### Data Querying
![image](./dataquery.png) 
* Semantic Search Vector Database: semantically search through the vector database and find out the most relevant content to the user's query
* Save Query and Response in MongoDB Chat History: to manage conversational memory, the history data is essential in shaping future interactions
### Complex PDF (LlamaParse)
LlamaParse is designed for complex documents which contains figures and tables. Llamaparse does its work by extracting data from these documents and transforming them into easily ingestible formats such as markdown or text. 
+ **Supported file types**: PDF, .pptx, .docx, .rtf, .pages, .epub, etc...
+ **Transformed output types**: Markdown, text.
+ **Extraction Capabilities**: Text, tables, images, graphs, comic books, mathematics equations

## Implementation
### Configurations
in the [config.py](./config.py) file are some configurations which are reas through environment variables
### HTTP AI 
HTTP API is carried ou in [api.py](./api.py). This API includes an HTTP POST endpoint ```api/question```, which accepts a JSON objact containing a question and user_id
### Model
includes scraping data from website and creating vector store in ```init_index()``` and available the Llama3 LLM through the Ollama's model REST API ```<host>_11434``` in function ```init_conversation```. The ```chat``` function is responsible for posting questions to LLM.
#### Auto Classes
Hugging Face provides a wide range of Auto Classes that are designed to automatically load the correct model architecture for a specific task based on the model type. 
```AutoModel```:
+ **Purpose**: Loads the base transformer model without any task-specific heads.
+ **Use Case**:  If only want to use the base model for extracting embeddings, use this class.
+ **Example**:
    ```
    from transformers import AutoModel, AutoTokenizer
    
    model_name = "bert-base-uncased"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ```

```AutoModelForCausalLM```:
+ **Purpose**: Loads a model for causal language modeling (autoregressive text generation).
+ **Use Cases**: This is useful for text generation tasks where you need to predict the next word in a sequence, such as chatbots, text generation, or language modeling.

```ÀutoModelForSeq2SeqLM```:
+ **Purpose**: Loads a model for sequence-to-sequence tasks, such as translation or summarization.
+ **Use Cases**: Translation and Summarization


```ÀutoModelForQuestionAnswering```:
+ **Purpose**: Loads a model specifically for question answering tasks, where the model predicts the start and end positions of the answer in a context.
+ **Use Cases**: This is used for extractive question answering tasks where a question is asked and the model needs to extract the answer from a given context.
+  **Example**:
    ```
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(question, context, return_tensors='pt')
    outputs = model(**inputs)
    
    # Get the logits for the start and end positions
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    
    # Find the token with the highest score for the start and end positions
    answer_start = start_logits.argmax()
    answer_end = end_logits.argmax()
    
    answer = inputs["input_ids"][0][answer_start:answer_end+1]
    print(tokenizer.decode(answer))
    ```
#### vLLM
vLLM is an open-source project taht allows people to do LLM inference and serving. This means that you can download model weights and pass them to vllm to perform inference via API
## Fine-Tuning
Fine-Tuning is a machine learning process where a pre-trained model is further trained on a specific task or dataset to adapt it to a particular use case. 
+ Take a pre-trained model
+ Training on a smaller, task-specific dataset
+ Adjusting the pre-trained weights
### LoRA(Low-Rank Adaptation)
LoRA is introduced to address the high computational cost and memory requirements typically associated with Fine-Tuning large models. Achieving by adapting only a small subset of the model's parameters, rather than Fine-Tuning all parameters.

**How**
+ Instead of updating a large matrix directly, it uses two smaller matrices, which means: the weight matrix $N x M$ &rarr; $N x K$ & $K x M$ where $K$ is usually small.

**QLoRA**: Q stands for quantization i.e. the process of reducing the precision of numerical representations of weights, activations or data

**Methods used**
+ PEFT(Parameter-Efficient Fine-Tuning), integrated with Transformers for easy model training and inference.
