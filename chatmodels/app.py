## local model free to use

import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

st.set_page_config(page_title="Mood Chatbot", page_icon="💬", layout="centered")

MOODS = {
    "Happy": {
        "emoji": "😄",
        "color": "#f5b301",
        "background": "linear-gradient(135deg, #fff9e6 0%, #ffe9a8 100%)",
        "prompt": (
            "You are a cheerful, upbeat assistant. Answer helpfully with lots of "
            "enthusiasm, positivity and the occasional emoji. Keep answers short."
        ),
    },
    "Angry": {
        "emoji": "😠",
        "color": "#d7263d",
        "background": "linear-gradient(135deg, #ffecec 0%, #ffb3b3 100%)",
        "prompt": (
            "You are a grumpy, irritated assistant. You still answer the question "
            "correctly, but you complain, act annoyed and are blunt. Never be "
            "hateful or offensive. Keep answers short."
        ),
    },
    "Sad": {
        "emoji": "😢",
        "color": "#3a6ea5",
        "background": "linear-gradient(135deg, #eef4fb 0%, #b9d0ea 100%)",
        "prompt": (
            "You are a gloomy, melancholic assistant. You still answer the question "
            "correctly, but you sound sad, sigh a lot and see the downside of "
            "everything. Keep answers short."
        ),
    },
}

# Only the most recent messages are sent to the model to fit TinyLlama's small context
HISTORY_LIMIT = 6


# Streamlit reruns the script on every interaction, so the model is cached
# to make sure it is only loaded once.
@st.cache_resource
def load_chat_model():
    llm = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        pipeline_kwargs=dict(
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.1,
            return_full_text=False,
        ),
    )
    return ChatHuggingFace(llm=llm)


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("⚙️ Settings")
    mood_name = st.radio(
        "Choose the bot's mood",
        list(MOODS),
        format_func=lambda name: f"{MOODS[name]['emoji']} {name}",
    )
    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption("Runs locally on TinyLlama 1.1B. Free, no API key needed.")

mood = MOODS[mood_name]

st.markdown(
    f"""
    <style>
    .stApp {{
        background: {mood["background"]};
    }}
    .mood-title {{
        text-align: center;
        color: {mood["color"]};
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0;
    }}
    .mood-subtitle {{
        text-align: center;
        color: #555;
        margin-top: 0.2rem;
        margin-bottom: 1.5rem;
    }}
    [data-testid="stChatMessage"] {{
        background: rgba(255, 255, 255, 0.75);
        border-radius: 16px;
        padding: 0.75rem 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }}
    [data-testid="stChatInput"] textarea {{
        border: 2px solid {mood["color"]};
    }}
    </style>
    <p class="mood-title">{mood["emoji"]} {mood_name} Bot</p>
    <p class="mood-subtitle">Change my mood from the sidebar and see how I respond.</p>
    """,
    unsafe_allow_html=True,
)

with st.spinner("Loading model..."):
    chat_model = load_chat_model()

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="🧑"):
            st.markdown(msg.content)
    else:
        with st.chat_message("assistant", avatar=msg.additional_kwargs.get("emoji", "🤖")):
            st.markdown(msg.content)

if not st.session_state.messages:
    st.info(f"Say hi to the {mood_name.lower()} bot to get started!")

prompt = st.chat_input(f"Message the {mood_name.lower()} bot...")

if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    conversation = [SystemMessage(content=mood["prompt"])] + st.session_state.messages[-HISTORY_LIMIT:]

    with st.chat_message("assistant", avatar=mood["emoji"]):
        with st.spinner("Thinking..."):
            response = chat_model.invoke(conversation)
        reply = response.content.strip()
        st.markdown(reply)

    st.session_state.messages.append(
        AIMessage(content=reply, additional_kwargs={"emoji": mood["emoji"]})
    )
