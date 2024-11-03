from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
import ssl


load_dotenv()

def initialize_neo4j_vector(index_name: str) -> tuple[Neo4jGraph, Neo4jVector]:
    # Initialisation de la connexion Neo4j
    url = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    keyword_index_name= f"{index_name}_keyword"
    # Initialiser le graph
    graph = Neo4jGraph(url=url, username=username, password=password)
    # Initialiser les embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
    )

    # Créer le store Neo4jVector en utilisant les index donnés
    store = Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=url,
        username=username,
        password=password,
        index_name=index_name,
        keyword_index_name=keyword_index_name,
        search_type="hybrid"
    )
    return graph, store

