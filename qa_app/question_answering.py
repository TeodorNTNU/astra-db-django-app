from langchain_astradb import AstraDBVectorStore
from langchain_openai import OpenAIEmbeddings
from astrapy import DataAPIClient
from django.conf import settings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from qa_app.prompt_template import PROMPT

# Initialize the Astra client
# client = DataAPIClient(settings.ASTRA_DB_APPLICATION_TOKEN)
# db = client.get_database_by_api_endpoint(
#     settings.ASTRA_DB_API_ENDPOINT,
#     namespace=settings.ASTRA_DB_KEYSPACE
# )

# Initialize the client
client = DataAPIClient('AstraCS:kLtbxENSOmsrPXgyuzuZNcEv:427e8ae4b2d998dca7b720ee235e1601a62fbf4d19dda0bd6de068c901fbb21b')
db = client.get_database_by_api_endpoint(
  "https://466c9033-8aae-49a6-812d-af55a2484c7e-us-east1.apps.astra.datastax.com",
    namespace="text_qa_keyspace",
)
      
print(f"Connected to Astra DB: {db.list_collection_names()}")

# Initialize the embeddings
embedding = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)

# Initialize the vector store
# vstore = AstraDBVectorStore(
#     embedding=embedding,
#     namespace=settings.ASTRA_DB_KEYSPACE,
#     collection_name=settings.ASTRA_DB,
#     token=settings.ASTRA_DB_APPLICATION_TOKEN,
#     api_endpoint=settings.ASTRA_DB_API_ENDPOINT,
# )

try:
    vstore = AstraDBVectorStore(
        embedding=embedding,
        namespace="text_qa_keyspace",
        collection_name='text_qa_collection',  # Correct collection name
        token='AstraCS:kLtbxENSOmsrPXgyuzuZNcEv:427e8ae4b2d998dca7b720ee235e1601a62fbf4d19dda0bd6de068c901fbb21b',
        api_endpoint="https://466c9033-8aae-49a6-812d-af55a2484c7e-us-east1.apps.astra.datastax.com",
    )
except Exception as e:
    print(f"Error initializing AstraDBVectorStore: {e}")


# gpt-3.5-turbo-16k == 16k token context window - easy to fit retrieved docs into Context for a query
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0, streaming=True)

def retriever(question, instructions='', k=3):
    partial_prompt = PROMPT.partial(instructions=instructions)
    retriever = vstore.as_retriever(search_kwargs={'k': k})

    if instructions:
        chain_type_kwargs = {'prompt': partial_prompt}
        qa_retriever = RetrievalQA.from_chain_type(
            llm=llm, 
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs=chain_type_kwargs
        )
    else:
        qa_retriever = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff"
        )

    return qa_retriever.invoke({"query": question, "instructions": instructions})

def get_documents():
    collection = db.get_collection(settings.ASTRA_DB)
    documents = collection.find()
    content_list = [doc['content'][:1000] for doc in documents if 'content' in doc]
    return content_list
