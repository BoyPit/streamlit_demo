import streamlit as st
import asyncio
from autogen import AssistantAgent, UserProxyAgent
import streamlit_authenticator as stauth
from autogen.token_count_utils import count_token # Load config
from streamlit_authenticator.utilities.hasher import Hasher


# Pre-hashing all plain text passwords once


authenticator = stauth.Authenticate(
    dict(st.secrets['credentials']),
    st.secrets['cookie']['name'],
    st.secrets['cookie']['key'],
    st.secrets['cookie']['expiry_days'],
)
with st.sidebar:
    st.write("# AutoGen Chat Agents")
    st.write(f'Welcome *{st.session_state["name"]}*')
    if st.session_state['authentication_status']:
       authenticator.logout()
if "data" not in st.session_state:
    st.session_state["data"] = 0  # Initial value

if "variation" not in st.session_state:
    st.session_state["variation"] = 0  # Initial value

col1, col2, col3 = st.columns(3)
col1.metric("Message token", st.session_state["data"], st.session_state["variation"])
col2.metric("AI Agent", 2)
col3.metric("Humidity", "86%", "4%")
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
        st.chat_message(msg["role"]).write(msg["content"])

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
        
        class TrackableAssistantAgent(AssistantAgent):
            def _process_received_message(self, message, sender, silent):
                print(message)
                st.session_state.messages.append({"role": "assistant", "content": message})
                st.chat_message("assistant").write(message)
                # Get the actual usage and print it
                return super()._process_received_message(message, sender, silent)
        
        class TrackableUserProxyAgent(UserProxyAgent):
            def _process_received_message(self, message, sender, silent):
                st.session_state.messages.append({"role": "user", "content": message})
                st.chat_message("user").write(message)
                print(count_token(message))
                size = count_token(message)
                st.session_state["variation"] = size - st.session_state["data"] 
                st.session_state["data"] = size
                st.rerun()

                return super()._process_received_message(message, sender, silent)
        
        st.session_state["assistant"] = TrackableAssistantAgent(name="assistant", llm_config=llm_config)
        st.session_state["user_proxy"] = TrackableUserProxyAgent(name="user", human_input_mode="NEVER", code_execution_config=False, max_consecutive_auto_reply=0)

        # Crée une boucle d'événement unique pour l'application Streamlit
        st.session_state["loop"] = asyncio.new_event_loop()
        asyncio.set_event_loop(st.session_state["loop"])

        # Fonction asynchrone pour initier la conversation
        async def initiate_chat():
            await st.session_state["user_proxy"].a_initiate_chat(
                st.session_state["assistant"],
                message="Bonjour",
            )
        # Exécute la conversation initiale
        st.session_state["loop"].run_until_complete(initiate_chat())

    # Fonction principale de Streamlit
    def main():
        if prompt := st.chat_input():
            # Envoie le message de l'utilisateur et récupère la réponse de l'assistant
            with st.spinner('Wait for it...'):
                st.session_state["user_proxy"].send(prompt, st.session_state["assistant"])

    if __name__ == "__main__":
        main()

elif st.session_state['authentication_status'] is False:
    st.error('Username/password is incorrect')
elif st.session_state['authentication_status'] is None:
    st.warning('Please enter your username and password')
