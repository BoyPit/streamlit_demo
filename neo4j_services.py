from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from typing_extensions import Annotated
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Tuple, List, Optional
import os
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from entities import Entities
from typing import List, Tuple, Any, Annotated



def generate_full_text_query(input: str) -> str:
    """
    Generate a full-text search query for a given input string.

    This function constructs a query string suitable for a full-text search.
    It processes the input string by splitting it into words and appending a
    similarity threshold (~2 changed characters) to each word, then combines
    them using the AND operator. Useful for mapping entities from user questions
    to database values, and allows for some misspelings.
    """
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()

# Full-text index query with the new entity structure
async def structured_retriever(question: str, graph) -> str:
    """
    Collects the neighborhood of entities mentioned
    in the question by querying the Neo4j graph database,
    using structured entity extraction.
    """
    result = ""
    # Few-shot examples to guide the model
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are extracting entities related to an industrial machine from the text. "
                "Map the entities to the following categories: machines, components, subsystems, "
                "characteristics, tools, and generic nodes.",
            ),
            (
                "human",
                "Here is an example of how to categorize the entities: "
                "'A quoi est connecté le relais K34U ?' -> components: ['relais K34U'], machines: [], subsystems: [], characteristics: [], tools: []\n"
                "'Quelle est la caractéristique du moteur M1 ?' -> components: ['moteur M1'], characteristics: ['caractéristique'], machines: [], subsystems: [], tools: []\n"
                "'Le système de contrôle de la machine a un problème.' -> components: ['système de contrôle'], machines: ['machine'], subsystems: [], characteristics: ['problème'], tools: []\n"
                "Now extract the entities from this question: {question}",
            ),
        ]
    )
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
    entity_chain = prompt | llm.with_structured_output(Entities)

    # Step 1: Retrieve entities from the question using entity_chain
    entities = entity_chain.invoke({"question": question})

    # Step 2: Define a function to handle querying for each entity type
    def query_graph_for_entity(entity: str, graph) -> str:
        # Execute the query using LangChain's graph.query to retrieve nodes and their neighbors
        response = graph.query(
            """
            CALL db.index.fulltext.queryNodes('LE06_index', $query, {limit:6})
            YIELD node, score
            WITH node, score
            MATCH (node)-[r:MENTIONS]->(neighbor)
            RETURN node AS source_node, r, neighbor, score

            UNION ALL

            CALL db.index.fulltext.queryNodes('LE06_index', $query, {limit:6})
            YIELD node, score
            WITH node, score
            MATCH (node)<-[r:MENTIONS]-(neighbor)
            RETURN node AS source_node, r, neighbor, score

            LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        
        # Build a hierarchical structure from the relationships
        hierarchy = {}

        for el in response:
            source_node = el["source_node"]
            neighbor = el["neighbor"]
            
            # Get node properties to create readable keys
            source_id = source_node.get("id", "N/A")
            source_props = ", ".join([f"{k}: {v}" for k, v in source_node.items()])
            source_key = f"Node {source_id} ({source_props})"
            
            neighbor_id = neighbor.get("id", "N/A")
            neighbor_props = ", ".join([f"{k}: {v}" for k, v in neighbor.items()])
            neighbor_key = f"Neighbor {neighbor_id} ({neighbor_props})"
            
            if source_key not in hierarchy:
                hierarchy[source_key] = {}
            
            # Create a structure with neighbor details under each source node
            hierarchy[source_key][neighbor_key] = f"Score: {el['score']}"
        
        # Function to recursively build the text representation
        def build_context(hierarchy_level, indent=0):
            text = ""
            for node_key, children in hierarchy_level.items():
                indentation = "  " * indent
                text += f"{indentation}- {node_key}\n"
                if isinstance(children, dict):
                    text += build_context(children, indent + 1)
                else:
                    text += f"{indentation}  {children}\n"  # Direct score display for leaf nodes
            return text

        # Build and return the context
        context = "Entity Structure:\n\n"
        context += build_context(hierarchy)
        return context

    
    # Step 3: Query for each type of entity extracted and append results to the final output
    if entities.machines:
        for machine in entities.machines:
            result += f"Context: Machine - {machine}\n"
            result += query_graph_for_entity(machine, graph)

    if entities.components:
        for component in entities.components:
            result += f"Context: Component - {component}\n"
            result += query_graph_for_entity(component, graph)
    
    if entities.subsystems:
        for subsystem in entities.subsystems:
            result += f"Context: Subsystem - {subsystem}\n"
            result += query_graph_for_entity(subsystem, graph)
    
    if entities.characteristics:
        for characteristic in entities.characteristics:
            result += f"Context: Characteristic - {characteristic}\n"
            result += query_graph_for_entity(characteristic, graph)
    
    if entities.tools:
        for tool in entities.tools:
            result += f"Context: Tool - {tool}\n"
            result += query_graph_for_entity(tool, graph)
    
    if entities.generic_nodes:
        for generic_node in entities.generic_nodes:
            result += f"Context: Generic Node - {generic_node}\n"
            result += query_graph_for_entity(generic_node, graph)
    
    return result


# Wrapper for transforming retriever function with store and graph as constants
def create_retriever(store, graph, score_threshold: float = 0.8, top_k: int = 2):
    async def retriever(question: Annotated[str, "Keyword to retrieve in the database"]) -> str:
        print(f"Received query: {question}")
        
        # Retrieve structured data from the knowledge graph
        structured_data = await structured_retriever(question, graph)
        
        if not structured_data:
            print("No structured data found. Proceeding with unstructured data only.")
            structured_data = "No relevant structured data found."

        # Retrieve unstructured data with relevance scores
        unstructured_data_with_scores: List[Tuple[Any, float]] = await store.asimilarity_search_with_relevance_scores(
            question, k=top_k, score_threshold=score_threshold
        )

        # Filter documents based on relevance score
        relevant_unstructured_data = [
            f"(Score: {score:.2f}) {doc.page_content}" for doc, score in unstructured_data_with_scores if score >= score_threshold
        ]

        # Format the response (without f-string for multiline)
        final_data = """
        Structured Data:
        {structured_data}

        Unstructured Data (Relevance Scores):
        {documents}
        """.format(
            structured_data=structured_data,
            documents="\n#Document ".join(relevant_unstructured_data)
        )
        
        return final_data.strip()
    
    return retriever


def fulltext_search_with_relationships(search_term : Annotated[str, "Key word to retrieve in the database"]) -> str:
    """
    Performs a full-text search across multiple node types using a full-text index.
    Retrieves the top 5 results, along with all relationships and related nodes for each matched node, including all node properties.
    
    Args:
        graph: Initialized Neo4jGraph object.
        search_term: The term or phrase to search for in the node names.
        
    Returns:
        A structured textual answer that includes:
            - The matched node and all its relationships.
            - The types of the relationships.
            - All properties of both the matched node and related nodes.
    """
        # Check if the search term is empty
    if not search_term or search_term.strip() == "":
        return "The search term is empty. Please provide a valid search term."

    # Define the query for full-text search across multiple node types, retrieving all relationships
    query = """
    CALL db.index.fulltext.queryNodes('allNodesNameIndex', $search_term) YIELD node, score
    MATCH (node)-[r]-(related)
    RETURN node, labels(node) AS node_labels, r AS relationship, related, labels(related) AS related_labels, score
    ORDER BY score DESC
    LIMIT 15
    """
    
    # Execute the query with the search_term as a parameter
    result = graph.query(query, {"search_term": search_term})
    
    # Initialize the structured answer
    structured_answer = ""
    
    # Dictionary to group all relationships for each node
    node_relations = {}

    # Iterate over the results to group all relationships per matched node
    for record in result:
        node = record['node']
        node_labels = record['node_labels']
        related_node = record['related']
        related_labels = record['related_labels']
        relationship = record['relationship']  # Relationship object returned by Neo4j
        score = record['score']
        
        # Get all properties of the matched node
        node_properties = ', '.join([f"{key}: {value}" for key, value in node.items()])
        
        # Get all properties of the related node
        related_node_properties = ', '.join([f"{key}: {value}" for key, value in related_node.items()])
        
        # Get the relationship type and properties
        if isinstance(relationship, tuple):
            relationship_type = relationship[0]  # First element in tuple could be relationship type
            relationship_properties = ', '.join([str(item) for item in relationship[1:]])  # Assume rest are properties
        else:
            # If it's not a tuple, handle it as an object (likely Neo4j relationship object)
            relationship_type = relationship.type
            relationship_properties = ', '.join([f"{key}: {value}" for key, value in relationship.items()])
        
        # Group relationships by the matched node
        node_key = f"Node type: {node_labels}, properties: [{node_properties}] and relevance score: {score}"
        if node_key not in node_relations:
            node_relations[node_key] = []
        
        node_relations[node_key].append(
            f"Is related to node type: {related_labels}, properties: [{related_node_properties}].\n"
            f"Relationship type: {relationship_type}, properties: [{relationship_properties}]."
        )
    
    # Build the final structured answer
    for node_key, relations in node_relations.items():
        structured_answer += f"\nMatched Node:\n{node_key}\n"
        for relation in relations:
            structured_answer += f"{relation}\n"
        structured_answer += "------------------------------------------\n"
    
    return structured_answer