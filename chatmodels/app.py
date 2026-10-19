## local model free to use

import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

st.set_page_config(page_title="Chatbot", page_icon="💬")


# Streamlit reruns the script on every interaction, so the model is cached
# to make sure it is only loaded once.
@st.cache_resource
def load_chat_model():
    llm = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        pipeline_kwargs=dict(
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
        ),
    )
    return ChatHuggingFace(llm=llm)


chat_model = load_chat_model()

# Keep the message history across reruns
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(
            content="You are a helpful assistant that can answer questions and help with tasks."
        ),
    ]
if "ended" not in st.session_state:
    st.session_state.ended = False

st.title("💬 Chatbot")
st.caption("Welcome to the chatbot. Type 0 to exit the application.")

# Render previous messages
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Exit state (equivalent of typing 0 in the console version)
if st.session_state.ended:
    st.info("Chat ended. Refresh the page to start a new chat.")
    st.stop()

prompt = st.chat_input("You:")

if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))

    if prompt == "0":
        st.session_state.ended = True
        st.rerun()

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_model.invoke(prompt)
        st.markdown(response.content)

    st.session_state.messages.append(AIMessage(content=response.content))