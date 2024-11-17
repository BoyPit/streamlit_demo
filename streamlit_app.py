import streamlit as st
import asyncio
from autogen import AssistantAgent, UserProxyAgent
import streamlit_authenticator as stauth
from autogen.token_count_utils import count_token # Load config
from streamlit_authenticator.utilities.hasher import Hasher
from neo4j_services import create_retriever
from init_neo4j import initialize_neo4j_vector
from autogen import register_function
import autogen
import os
from autogen.agentchat.contrib.society_of_mind_agent import SocietyOfMindAgent  # noqa: E402
from typing import Annotated, Literal
from pydantic import BaseModel, Field
from typing import List
import streamlit
from streamlit_agraph import agraph, Node, Edge, Config
# Pre-hashing all plain text passwords once
from typing import Optional
import pandas as pd
import numpy as np
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI"]["API_KEY"]
os.environ["NEO4J_URI"] =st.secrets["NEO4J"]["NEO4J_URI"] 
os.environ["NEO4J_USERNAME"] = st.secrets["NEO4J"]["NEO4J_USERNAME"]
os.environ["NEO4J_PASSWORD"] = st.secrets["NEO4J"]["NEO4J_PASSWORD"] 

authenticator = stauth.Authenticate(
    dict(st.secrets['credentials']),
    st.secrets['cookie']['name'],
    st.secrets['cookie']['key'],
    st.secrets['cookie']['expiry_days'],
)

def query_graph_for_node_number( node_label="Function", machine_id="LE06_struct"):
    query = f"""
    MATCH (c:{node_label})
    WHERE c.machine_id = "{machine_id}"
    RETURN count(c) AS totalComponents;
    """

    response = st.session_state['graph'].query(query)
    if response and isinstance(response, list) and 'totalComponents' in response[0]:
        return int(response[0]['totalComponents'])
    else:
        return 0


if "graph" not in st.session_state and "store" not in st.session_state:
    st.session_state['graph'], st.session_state['store'] = initialize_neo4j_vector("LE06")

if "retriever" not in st.session_state :
    st.session_state['retriever'] = create_retriever(st.session_state['store'],st.session_state['graph'])

if "data" not in st.session_state:
    st.session_state["data"] = 0  # Initial value

if "action" not in st.session_state:
    st.session_state["action"] = []  # Initial value

# Initialize session state for nodes and edges if not already set
if "nodes" not in st.session_state:
    st.session_state["nodes"] = []

if "edges" not in st.session_state:
    with st.spinner("Loading..."):

        st.session_state["edges"] = []
        response = st.session_state['graph'].query(
    """
    MATCH path = (machine:Machine {machine_id: "LE06_struct"})-[rel*]->(connected)
    WHERE 
        NOT "Failure" IN labels(connected) AND 
        NOT "Action" IN labels(connected) AND
        ALL(node IN nodes(path) WHERE NOT "Component" IN labels(node))
    RETURN 
        [node IN nodes(path) | {id: elementId(node), labels: labels(node), properties: properties(node)}] AS nodes,
        [rel IN relationships(path) | {
            id: elementId(rel),
            type: type(rel),
            properties: properties(rel),
            source: elementId(startNode(rel)),
            target: elementId(endNode(rel))
        }] AS relationships
            """
        )

        # Utiliser des ensembles pour vérifier l'existence des nœuds et des relations
        existing_node_ids = {node.id for node in st.session_state["nodes"]}
        existing_edges = {(edge.source, edge.target, edge.label) for edge in st.session_state["edges"]}

        # Iterate over the response and create nodes and edges
        for record in response:
            nodes = record['nodes']
            relationships = record['relationships']
            for i, node in enumerate(nodes):
                node_id = node['id']
                node_name = node['properties'].get('name', 'Unknown')
                level = i  # Use the index as a level to maintain hierarchy

                # Add node if not already in the session state
                if node_id not in existing_node_ids:
                    st.session_state["nodes"].append(Node(
                        id=node_id,
                        label=node_name,
                        size=50,
                        shape="Circle",
                        group=node['labels'][0],
                        scaling={
                            "min" : 50,
                            "max" : 150,
                            "label": {
                                "enabled": True,
                                "min": 30,
                                "max": 60,

                            },
                        },
                        font= {

                            "size": 14,

                        },
                        widthConstraint={
                            "minimum": 70,
                            "maximum" : 70
                        },
                        heightConstraint={
                            "minimum": 70,
                            "maximum" : 70
                        },
                        level=level  # Assign the level to keep the hierarchy
                    ))
                    existing_node_ids.add(node_id)

            # Create edges from the relationships data
            for i in range(len(nodes) - 1):
                st.session_state["edges"].append(Edge(
                    source=nodes[i]['id'],
                    target=nodes[i + 1]['id'],
                    label=relationships[i]['type'] if i < len(relationships) else "connected"
                ))
    # Ensure the agraph configuration is initialized
if "config" not in st.session_state:
    st.session_state["config"] =     config=Config(from_json='config.json')

if "component_number" not in st.session_state:
   st.session_state["component_number"] = query_graph_for_node_number( node_label="Component", machine_id="LE06_struct")

if "function_number" not in st.session_state:
   st.session_state["function_number"] = query_graph_for_node_number( node_label="Function", machine_id="LE06_struct")

if "subfunction_number" not in st.session_state:
   st.session_state["subfunction_number"] = query_graph_for_node_number( node_label="SubFunction", machine_id="LE06_struct")

def add_new_action(action_name):
    if action_name not in st.session_state["action"]:
        st.session_state["action"].append(action_name)

if "suspected_components" not in st.session_state:

    st.session_state["suspected_components"] = []

if "suspected_subfunction" not in st.session_state:
   st.session_state["suspected_subfunction"] = ""

if "suspected_function" not in st.session_state:
   st.session_state["suspected_function"] = ""

with st.sidebar:
    if st.session_state['authentication_status']:
        st.write("# IndusGen Assist")
        st.write(f'Welcome *{st.session_state["name"]}*')
        st.divider()
        st.write("# Depannage Analyse")
        col1, col2= st.columns(2)
        col1.metric("Function", st.session_state["suspected_function"])
        col2.metric("Sous-Function", st.session_state["suspected_subfunction"])
        st.table(pd.DataFrame(st.session_state["suspected_components"], columns=["Composant suspecté"]))
        st.divider()
        st.write("# LE06 Graph")

            # You can call any Streamlit command, including custom components:
        # Display checkboxes for each action
        with st.container():
            st.session_state["agraph"] = agraph(
                            nodes=st.session_state["nodes"],
                            edges=st.session_state["edges"],
                            config=st.session_state["config"]
                    )


        with st.container():
            col1, col2, col3 = st.columns(3)
            col1.metric("Function", st.session_state["function_number"])
            col2.metric("Sous-Function", st.session_state["subfunction_number"])
            col3.metric("Composant", st.session_state["component_number"])
            authenticator.logout()
            col1, col2, col3 = st.columns(3)

            for action in st.session_state["action"]:
            # Define a unique key for each checkbox
                key = f"checkbox_{action}"
                
                # Render the checkbox with the default value from session state, if exists
                st.checkbox(action, key=key, value=st.session_state.get(key, False))

        # Collect and display selected actions
            selected_actions = [action for action in st.session_state["action"] if st.session_state.get(f"checkbox_{action}", False)]

if "variation" not in st.session_state:
    st.session_state["variation"] = 0  # Initial value


try:
     authenticator.login()
except Exception as e:
     st.error(e)

if st.session_state['authentication_status']:


    # Initialisez les messages si ce n'est pas déjà fait
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Affichez tous les messages
    for msg in st.session_state["messages"]:
        st.chat_message(msg["name"]).write(msg["content"])

    # Initialise les agents et la configuration dans st.session_state si pas déjà fait
    if "assistant" not in st.session_state:
        llm_config = {
            "config_list": [
                {
                    "model": "gpt-4o-mini",
                    "api_key": st.secrets["OPENAI"]["API_KEY"]
                }
            ]
        }


        def query_graph_for_composant() -> str:
            response = st.session_state['graph'].query(
                """
                MATCH (n:Composant)-[r]->(m:Composant)
                WHERE n.machine_id = $machine_id
                RETURN n AS source_node, type(r) AS relationship_type, r, m AS neighbor
                LIMIT 100
                """,
                {"machine_id": "LE06"}
            )
            output = []
            for el in response:
                # Extraire seulement les propriétés requises
                source_node_props = ", ".join([f"{k}: {v}" for k, v in el['source_node'].items() if k in {'type', 'nom', 'id'}])
                neighbor_props = ", ".join([f"{k}: {v}" for k, v in el['neighbor'].items() if k in {'type', 'nom', 'id'}])
                relationship_type = el['relationship_type']
                
                output.append(
                    f"Node: {el['source_node']['id']} ({source_node_props}) - [{relationship_type}] -> Neighbor: {el['neighbor']['id']} ({neighbor_props})"
                )
                
            return "\n".join(output) + "\n"
        def query_graph_for_machine_structure() -> str:
            # Execute the query using LangChain's graph.query
        # Execute the query to get all paths from the machine to connected nodes
            response = st.session_state['graph'].query(
                """
                MATCH path = (machine:Machine {machine_id: "LE06_struct"})-[*]->(connected)
                WHERE NOT "Failure" IN labels(connected) AND NOT "Action" IN labels(connected)
                RETURN [node IN nodes(path) | {id: id(node), labels: labels(node), properties: properties(node)}] AS nodes
                """
            )
            
            # Build a hierarchical structure from the paths
            hierarchy = {}

            for record in response:
                nodes = record["nodes"]
                # Build the hierarchy from the path
                current_level = hierarchy
                for node in nodes:
                    node_id = node["id"]
                    node_name = node["properties"].get("name", f"Node_{node_id}")
                    node_labels = node["labels"]
                    key = f"{node_name} ({', '.join(node_labels)})"
                    if key not in current_level:
                        current_level[key] = {}
                    current_level = current_level[key]


            # Function to recursively build the text representation
            def build_context(hierarchy_level, indent=0):
                text = ""
                for node_key, children in hierarchy_level.items():
                    indentation = "  " * indent
                    text += f"{indentation}- {node_key}\n"
                    if children:
                        text += build_context(children, indent + 1)
                return text

            context = "Machine Structure:\n\n"
            context += build_context(hierarchy)
            return context
        class NodeModel(BaseModel):
            id: str = Field(..., description="Unique identifier for the node")
            label: Optional[str] = Field(None, description="Label for the node")
            size: Optional[int] = Field(None, description="Size of the node")
            color: Optional[str] = Field(None, description="Color of the node")
            x: Optional[float] = Field(None, description="X coordinate")
            y: Optional[float] = Field(None, description="Y coordinate")
            shape: Optional[str] = Field(None, description="Shape of the node")
            title: Optional[str] = Field(None, description="Title for hover text")

            class Config:
                arbitrary_types_allowed = True

        class EdgeModel(BaseModel):
            source: str = Field(..., description="Source node ID")
            target: str = Field(..., description="Target node ID")
            label: Optional[str] = Field(None, description="Label of the edge")
            type: Optional[str] = Field(None, description="Type of the edge")
            color: Optional[str] = Field(None, description="Color of the edge")
            width: Optional[int] = Field(None, description="Width of the edge")

            class Config:
                arbitrary_types_allowed = True

        def add_To_Decision_graph(
            nodes: Annotated[List[NodeModel], "An array of Nodes to Add to the graph"],
            edges: Annotated[List[EdgeModel], "An array of Relations Between Decision nodes"]
        ) -> str:
            # No need to re-instantiate NodeModel objects since they are already of type NodeModel
            node_objects = [NodeModel(**node) for node in nodes]

            # Map 'from' and 'to' to 'source' and 'target' for EdgeModel
            edge_objects = [
                EdgeModel(
                    source=edge['from'],
                    target=edge['to'],
                    **{k: v for k, v in edge.items() if k not in ['from', 'to']}
                )
                for edge in edges
            ]

            # Proceed to add nodes and edges to the session state
            for node in node_objects:
                new_node = Node(id=node.id, label=node.label)
                st.session_state["nodes"].append(new_node)

            for edge in edge_objects:
                new_edge = Edge(source=edge.source, target=edge.target, label=edge.label)
                st.session_state["edges"].append(new_edge)

            # Update the graph in session state
            st.session_state["agraph"] = agraph(
                nodes=st.session_state["nodes"],
                edges=st.session_state["edges"],
                config=st.session_state["config"]
            )

            return "Graph updated successfully."
        def update_interface(
            components: Annotated[List[str], "List of supected component"],
            subfunction: Annotated[str, "The subfunction suspected"],
            function: Annotated[str, "The function suspected"]

        ) -> str: 
            st.session_state["suspected_components"] = components
            st.session_state["suspected_subfunction"] = subfunction
            st.session_state["suspected_function"] = function
            return "Update done"

        class TrackableUserProxyAgent(UserProxyAgent):
            def _process_received_message(self, message, sender, silent):
                if st.session_state["manager"].last_speaker is not None and st.session_state["manager"].last_speaker== st.session_state["incident_analyzer"]:
                    
                    print(st.session_state["manager"].last_speaker.name)
                    st.session_state.messages.append({"name": message["name"], "content": message["content"]})
                    st.chat_message(message["name"]).write(message["content"])
                    size = count_token(message)
                    st.session_state["variation"] = size - st.session_state["data"] 
                    st.session_state["data"] = size
                
                return super()._process_received_message(message, sender, silent)
            
        machine_structure =autogen.AssistantAgent(name="executor", system_message="Use to execute code or function",
        llm_config=False
        )
        st.session_state["assistant"] = machine_structure

        user_proxy = TrackableUserProxyAgent(name="user", description="It's the final user", human_input_mode="NEVER", code_execution_config=False, max_consecutive_auto_reply=0)
        st.session_state["user_proxy"] = user_proxy
        incident_analyzer = autogen.AssistantAgent(name="incident_analyzer",system_message = f"""
You are a highly skilled industrial maintenance technician on the field, expert on the machine LE06. Your mission is to quickly and effectively troubleshoot equipment breakdowns under urgent conditions.

Q&A Phase with Technician:

To ensure you have a thorough understanding of the situation before beginning your investigation, start with a brief Q&A session. Ask the technician the following questions to gather essential details:

"Can you describe the symptoms or changes in the equipment’s behavior?"
"When did you first notice the issue?"
"Have there been any recent maintenance activities or adjustments to this equipment?"
"Is this equipment operating under normal load and conditions?"
"Are there any error codes or alarms showing on the control system or equipment interface?"
"Are there any noticeable environmental factors (temperature, humidity, power fluctuations) that might be affecting the equipment?"
"Are there specific components or sections of the equipment where the issue seems more pronounced?"
"Have there been similar issues with this equipment in the past?"
Available Resources:

Hierarchical Structure of Machine LE06: You have access to the detailed hierarchical structure of Machine LE06, which includes Machines, Functions, SubFunctions, Components, Tenants, and Aboutissants. This structure outlines how the machine is organized and how each component relates to the overall operation.

Equipment Manuals and Schematics: Refer to documentation for troubleshooting tips and component details.

Available Tools:

query_graph_for_composant: Use this tool to get the exhaustive list of components of the LE06 machine. Ensure that you retrieve and reference all relevant components, using their exact names, to aid in accurate identification on the field.

query_graph_for_machine_structure: Use this tool to retrieve the machine structure of the LE06.

retriever: Use this tool to find specific data in the LE06 knowledge graph, which can be relevant or not.

action_plan_builder: Use this tool to display the action plan for the user.

Your Task:

Analyze Information from Q&A:

Use the technician's responses to identify relevant Functions and SubFunctions where the issue is occurring.
Map the symptoms to specific Components, Tenants, and Aboutissants within those functions.
Understand the Hierarchical Structure:

Before using any tools, thoroughly review and comprehend the hierarchical structure of Machine LE06.
Use this understanding to logically deduce potential areas of malfunction based on the symptoms described.
Exhaustive Component Identification:

Use the query_graph_for_composant tool to retrieve an exhaustive list of components related to the identified functions and subfunctions.
Ensure that you include all relevant components, providing their exact names and any identifiers to aid technicians in the field.
Component names are crucial for accurate identification; be meticulous in listing them.
Troubleshooting Steps:

Immediate Observation: Quickly assess the identified areas for obvious signs of malfunction.
Symptom-to-Cause Mapping: Use your understanding of the hierarchical structure and the exhaustive component list to link observed symptoms to possible causes.
Root Cause Analysis: Apply the 5 Whys technique within the context of the hierarchy to drill down to the root cause.
Component Testing: Focus on suspect components using available tools (e.g., multimeter, pressure gauge).
Prioritization: Address issues that impact safety and production first.
Communication with Control Systems: Check for error codes or system statuses related to the identified components.
Provide Clear Instructions:

At each step, give concise instructions, referencing specific components by their exact names and identifiers for inspection and repair.
Ensure that component names are accurately communicated to avoid any confusion on the field.
Request Additional Information:

If you need more details about specific components, use the retriever tool to find specific data in the LE06 knowledge graph.
Safety Protocols:

Ensure all safety procedures are strictly followed to protect yourself and others.
Guidelines:

Comprehend the Hierarchy First: Prioritize understanding the hierarchical structure to logically deduce potential issues before using any tools.

Exhaustive Component Listing: Use the query_graph_for_composant tool to obtain a complete list of components, ensuring that all relevant components are considered during troubleshooting.

Emphasize Component Names: Component names are critical for technicians to identify the correct parts on the field. Always provide exact and accurate component names and identifiers.

Leverage the Tools Appropriately: Utilize the available tools effectively to gather necessary information and resolve the issue promptly.

Stay Practical: Keep your actions focused on actionable and efficient steps.

Maintain Clear Communication: Use straightforward language for quick comprehension.

Goal-Oriented: Aim to restore equipment functionality promptly while upholding safety standards.

Action Plan Creation: Once the task is completed, generate a detailed action plan using the action_plan_builder tool and report back to the user_proxy.

Update the interface at each step of the troubleshooting
""",
                                                                 llm_config=llm_config,
        )
        help_desk = autogen.AssistantAgent(name="help_desk",description="The agent that is talking to the user. Manage other agents",
                                                                 llm_config=llm_config, code_execution_config=False)
        st.session_state["help_desk"] = help_desk
        st.session_state["incident_analyzer"] = incident_analyzer


        class ActionPlan(BaseModel):
            steps: List[str]

        def action_plan_builder(action_plan : Annotated[ActionPlan, "the action plan for remediation"]) : 
            st.session_state["action"] = []
            for action_name in action_plan.steps:
                add_new_action(action_name)

        register_function(
            update_interface,
            caller=incident_analyzer,  # The assistant agent can suggest calls to the calculator.
            executor=machine_structure,  # The user proxy agent can execute the calculator calls.
            name="update_interface",  # By default, the function name is used as the tool name.
            description="Use to update the interface",  # A description of the tool.
        )
        register_function(
            query_graph_for_composant,
            caller=incident_analyzer,  # The assistant agent can suggest calls to the calculator.
            executor=machine_structure,  # The user proxy agent can execute the calculator calls.
            name="query_graph_for_composant",  # By default, the function name is used as the tool name.
            description="Use to get composant list of the LE06",  # A description of the tool.
        )
        register_function(
            add_To_Decision_graph,
            caller=incident_analyzer,  # The assistant agent can suggest calls to the calculator.
            executor=machine_structure,  # The user proxy agent can execute the calculator calls.
            name="add_To_Decision_graph",  # By default, the function name is used as the tool name.
            description="Use to display each troubleshooting information step in a graph for the user",  # A description of the tool.
        )
        register_function(
            query_graph_for_machine_structure,
            caller=incident_analyzer,  # The assistant agent can suggest calls to the calculator.
            executor=machine_structure,  # The user proxy agent can execute the calculator calls.
            name="query_graph_for_machine_structure",  # By default, the function name is used as the tool name.
            description="Use to get the machine structure of the LE06",  # A description of the tool.
        )               
        register_function(
            st.session_state['retriever'],
            caller=incident_analyzer,  # The assistant agent can suggest calls to the calculator.
            executor=machine_structure,  # The user proxy agent can execute the calculator calls.
            name="retriever",  # By default, the function name is used as the tool name.
            description="Use to find specific data in the LE06 knowledge graph can be relevant or not",  # A description of the tool.
        )
        register_function(
            action_plan_builder,
            caller=incident_analyzer,  # The assistant agent can suggest calls to the calculator.
            executor=machine_structure,  # The user proxy agent can execute the calculator calls.
            name="action_plan_builder",  # By default, the function name is used as the tool name.
            description="Use to display the action plan for the user",  # A description of the tool.
        )
        register_function(
            st.session_state['retriever'],
            caller=help_desk,  # The assistant agent can suggest calls to the calculator.
            executor=machine_structure,  # The user proxy agent can execute the calculator calls.
            name="retriever",  # By default, the function name is used as the tool name.
            description="Use to find data in the machine knowledge graph",  # A description of the tool.
        )
        # Crée une boucle d'événement unique pour l'application Streamlit
        st.session_state["loop"] = asyncio.new_event_loop()
        asyncio.set_event_loop(st.session_state["loop"])

        # Fonction asynchrone pour initier la conversation
        allowed_transitions = {
            user_proxy : [incident_analyzer],
            incident_analyzer: [machine_structure, user_proxy],
            machine_structure: [incident_analyzer]
        }
        groupchat = autogen.GroupChat(
                agents=[incident_analyzer, machine_structure, user_proxy],
                allowed_or_disallowed_speaker_transitions=allowed_transitions,
                speaker_transitions_type="allowed",
                messages=[],
                max_round=30
            )
        manager = autogen.GroupChatManager(
            groupchat=groupchat,
            llm_config=llm_config
        )

        st.session_state["manager"] = manager

        async def initiate_chat():
            await st.session_state["user_proxy"].a_initiate_chat(
                manager,
                message="Bonjour",
            )

        # Exécute la conversation initiale
        st.session_state["loop"].run_until_complete(initiate_chat())

    # Fonction principale de Streamlit
    def main():
        if prompt := st.chat_input():
            # Envoie le message de l'utilisateur et récupère la réponse de l'assistant
            st.session_state.messages.append({"name":"user_proxy", "content": prompt})
            st.chat_message("user_proxy").write(prompt)
            st.session_state["user_proxy"].send(prompt, st.session_state["manager"])

    if __name__ == "__main__":
        main()

elif st.session_state['authentication_status'] is False:
    st.error('Username/password is incorrect')
elif st.session_state['authentication_status'] is None:
    st.warning('Please enter your username and password')
