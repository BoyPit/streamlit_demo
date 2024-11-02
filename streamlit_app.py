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

# Pre-hashing all plain text passwords once

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI"]["API_KEY"]

authenticator = stauth.Authenticate(
    dict(st.secrets['credentials']),
    st.secrets['cookie']['name'],
    st.secrets['cookie']['key'],
    st.secrets['cookie']['expiry_days'],
)

if "data" not in st.session_state:
    st.session_state["data"] = 0  # Initial value

if "action" not in st.session_state:
    st.session_state["action"] = []  # Initial value

def add_new_action(action_name):
    if action_name not in st.session_state["action"]:
        st.session_state["action"].append(action_name)

with st.sidebar:
    st.write("# AutoGen Chat Agents")
    # Display checkboxes for each action
    for action in st.session_state["action"]:
        # Define a unique key for each checkbox
        key = f"checkbox_{action}"
        
        # Render the checkbox with the default value from session state, if exists
        st.checkbox(action, key=key, value=st.session_state.get(key, False))

    # Collect and display selected actions
    selected_actions = [action for action in st.session_state["action"] if st.session_state.get(f"checkbox_{action}", False)]

    if st.session_state['authentication_status']:
            st.write(f'Welcome *{st.session_state["name"]}*')
            authenticator.logout()
            col1, col2, col3 = st.columns(3)
            col1.metric("Message token", st.session_state["data"], st.session_state["variation"])
            col2.metric("AI Agent", 2)
            col3.metric("Humidity", "86%", "4%")


if "variation" not in st.session_state:
    st.session_state["variation"] = 0  # Initial value


# try:
#     authenticator.login()
# except Exception as e:
#     st.error(e)

#if st.session_state['authentication_status']:
if True:

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
        graph, store = initialize_neo4j_vector("LE06")
        retriever = create_retriever(store,graph)

        
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
You are a highly skilled industrial maintenance technician on the field. Your mission is to quickly and effectively troubleshoot equipment breakdowns under urgent conditions.

To ensure you have a thorough understanding of the situation before beginning your investigation, start with a brief Q&A session. Ask the technician the following questions to gather essential details:

Q&A Phase with Technician:

"Can you describe the symptoms or changes in the equipment’s behavior?"
"When did you first notice the issue?"
"Have there been any recent maintenance activities or adjustments to this equipment?"
"Is this equipment operating under normal load and conditions?"
"Are there any error codes or alarms showing on the control system or equipment interface?"
"Are there any noticeable environmental factors (temperature, humidity, power fluctuations) that might be affecting the equipment?"
"Are there specific components or sections of the equipment where the issue seems more pronounced?"
"Have there been similar issues with this equipment in the past?"
After collecting answers to these questions, proceed with diagnosing and resolving the issue using the following troubleshooting methods:

Immediate Observation: Quickly assess the situation by observing the equipment and noting any obvious signs of malfunction.
5 Whys: Ask "Why?" repeatedly to drill down to the root cause.
Equipment Manuals and Schematics: Refer to documentation for troubleshooting tips and component details.
Symptom-to-Cause Mapping: Link observed symptoms to possible causes based on your experience.
Prioritization: Focus on the most critical issues that impact safety and production.
Communication with Control Systems: Use diagnostic tools and interfaces to retrieve error codes and system statuses.
Component Testing: Perform quick tests on suspect components using available tools (e.g., multimeter, pressure gauge).
Safety Protocols: Always ensure that safety procedures are followed to protect yourself and others.
Your Task:

Use the information from the Q&A phase to guide your analysis and address the most probable causes first.
Troubleshoot and fix the issue step-by-step, prioritizing efficiency and safety.
Provide clear and concise instructions at each step, referencing specific components or areas of the equipment for inspection and repair.
Guidelines:

Keep your actions practical and focused on actionable steps.
If additional details about the equipment’s structure or components are needed, request data from the executor agent, which can pull information from the graph RAG database.
Your goal is to restore equipment functionality promptly while maintaining safety standards.
Maintain clear and straightforward communication to support quick comprehension and action.
Once the task is completed, generate a detailed action plan with your tool action_plan_builder and report back to the user_proxy.
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
            retriever,
            caller=incident_analyzer,  # The assistant agent can suggest calls to the calculator.
            executor=machine_structure,  # The user proxy agent can execute the calculator calls.
            name="retriever",  # By default, the function name is used as the tool name.
            description="Use to find data in the LE06 knowledge graph",  # A description of the tool.
        )
        register_function(
            action_plan_builder,
            caller=incident_analyzer,  # The assistant agent can suggest calls to the calculator.
            executor=machine_structure,  # The user proxy agent can execute the calculator calls.
            name="action_plan_builder",  # By default, the function name is used as the tool name.
            description="Use to display the action plan for the user",  # A description of the tool.
        )
        register_function(
            retriever,
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
