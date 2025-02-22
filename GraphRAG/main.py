# Load all the necessary packages and libraries
import os 
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.vectorstores import Neo4jVector
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.runnables import  RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from typing import List
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import context_precision, faithfulness, ResponseRelevancy, FactualCorrectness
from yfiles_jupyter_graphs import GraphWidget
from neo4j import GraphDatabase
from huggingface_hub import login
from huggingface_hub import InferenceClient
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
from datasets import Dataset
import json
import warnings
load_dotenv()

# login()

# Load LLM and embeddings model
model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

# llm = ChatOllama(temperature=0.6, model="qwen2")
llm = ChatOpenAI(temperature = 0.6, model = "gpt-3.5-turbo-0125")

os.environ['NEO4J_URI'] = 'neo4j+s://99597476.databases.neo4j.io'
os.environ['NEO4J_USERNAME'] = 'neo4j'
os.environ['NEO4J_PASSWORD'] = '7Io74oHv7d1zf4SPYNu7ZJQZc19lMdPFG4_z-clwHwE'
os.environ['NEO4J_DATABASE'] = 'neo4j'
os.environ['AURA_INSTANCEID'] = '99597476'
os.environ['AURA_INSTANCENAME'] = 'Instance01'

graph = Neo4jGraph()

import pickle

# Index based on existing graph
index_name = "vector"
keyword_index_name = "keyword"

vector_index = Neo4jVector.from_existing_index(
    model,
    index_name = index_name,
    keyword_index_name=keyword_index_name,
    search_type="hybrid"
)

retriever = vector_index.as_retriever()

graph.query("""
  SHOW VECTOR INDEXES
  """
)

graph.refresh_schema()

class Entities(BaseModel):
    """Identifying information about entities."""

    names: list[str] = Field(
        ...,
        description="All the person, organization, or business entities that "
        "appear in the text",
    )
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting organization and person entities from the text.",
        ),
        (
            "human",
            "Use the given format to extract information from the following "
            "input: {question}",
        ),
    ]
)
# Entities_converted = convert_to_openai_function(Entities)
structured_llm = llm.with_structured_output(Entities)
entity_chain = prompt | structured_llm

def generate_full_text_query(input: str) -> str:
    """
    Generate a full-text search query for a given input string.
    
    This function constructs a query string suitable for a full-text search.
    It processes the input string by splitting it into words and combines them
    using the AND operator, ensuring exact matches without allowing misspellings.
    """
    words = [el for el in remove_lucene_chars(input).split() if el]
    if not words:
        return ""
    full_text_query = " AND ".join(words)
    print(f"Generated Query: {full_text_query}")
    return full_text_query.strip()


# Fulltext index query
def graph_retriever(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned
    in the question
    """
    if not any(index["name"] == "entity" for index in graph.query("SHOW INDEXES")):
        graph.query("CREATE FULLTEXT INDEX entity FOR (n:__Entity__) ON EACH [n.id]")
    result = ""
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        response = graph.query(
            """
           CALL db.index.fulltext.queryNodes('entity', $query)
           YIELD node, score
           CALL {
            WITH node
            MATCH (node)-[r:!MENTIONS]->(neighbor)
            RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
            UNION ALL
            WITH node
            MATCH (node)<-[r:!MENTIONS]-(neighbor)
            RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
            } WITH output
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el['output'] for el in response])
    return result

def full_retriever(question: str):
    graph_data = graph_retriever(question)
    vector_data = [el.page_content for el in retriever.invoke(question)]
    final_data = f"""Graph data:
{graph_data}
vector data:
{"#Document". join(vector_data)}
    """
    return final_data

# from langchain_core.runnables import  RunnablePassthrough

# parser = StrOutputParser()

# template = """
# Answer the question based only on the following context: {context}

# Question: {question}
# - You are a representative at Western Sydney University, your job is to provide accurate information about \
# Western Sydney University to international students who wish to enrol.
# - Use natural language.
# - Perform formatting evaluation of the generated response. For example, If the query looks like it requires a list, \
# format the response to be a list. Make sure that you don't miss any components while listing.
# - For questions that requires simple retrieval, your answer should be concise and informative enough. The entities extracted \
# from the query should match exactly with the retrieved information. 
# - For example, you should return the exact match of subjects within a particular degree, do not take subjects that don't belong \
# to the degree mentioned in the query.
# - For question that requires extensive reasoning, your answers must not be longer than 4 paragraphs, try to connect to \
# neighbor entities for more contextually aware response.
# - If there's no entity matching the query, return "There is no information in the database regarding what \
# you are requesting. Perhaps you should check your spelling or try a different prompt."

# Answer:"""
# prompt = ChatPromptTemplate.from_template(template)

# chain = (
#         {
#             "context": full_retriever,
#             "question": RunnablePassthrough(),
#         }
#     | prompt
#     | llm
#     | parser
# )
 
# query = str(input("Enter your query: "))
# # print(graph_retriever(query))
# answer = (chain.invoke(input = query))
# # vector_store = InMemoryVectorStore.from_documents(pages, model)
# # retrieved_docs = vector_store.similarity_search(query, k=2)

# def full_answer():
#     print(answer)
#     # print("-" * 40)  # Adds a line of dashes as a separator 
#     # for doc in retrieved_docs:
#     #     print(f'Page {doc.metadata["page"]}: {doc.page_content[:300]}\n')

# full_answer()