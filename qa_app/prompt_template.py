from langchain.prompts import PromptTemplate

prompt_template_text = """
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}

Instructions: {instructions}:
"""

PROMPT = PromptTemplate(
    template=prompt_template_text, 
    input_variables=["context", "question", "instructions"]
)
