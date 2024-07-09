from langchain_astradb import AstraDBVectorStore
from langchain_openai import OpenAIEmbeddings
from astrapy import DataAPIClient
from django.conf import settings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from qa_app.prompt_template import PROMPT

# Initialize the Astra client
client = DataAPIClient(settings.ASTRA_DB_APPLICATION_TOKEN)
db = client.get_database_by_api_endpoint(
    settings.ASTRA_DB_API_ENDPOINT,
    namespace=settings.ASTRA_DB_KEYSPACE
)

# Initialize the embeddings
embedding = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)

# Initialize the vector store
vstore = AstraDBVectorStore(
    embedding=embedding,
    namespace=settings.ASTRA_DB_KEYSPACE,
    collection_name='text_qa_collection',
    token=settings.ASTRA_DB_APPLICATION_TOKEN,
    api_endpoint=settings.ASTRA_DB_API_ENDPOINT,
)

# gpt-3.5-turbo-16k == 16k token context window - easy to fit retrieved docs into Context for a query
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125")

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
    collection = db.get_collection("text_qa_collection")
    documents = collection.find()
    content_list = [doc['content'][:1000] for doc in documents if 'content' in doc]
    return content_list
