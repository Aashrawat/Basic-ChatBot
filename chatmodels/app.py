## local model free to use

import html
import time

import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

st.set_page_config(page_title="Mood Chatbot", page_icon="💬", layout="centered")

MOODS = {
    "Happy": {
        "emoji": "😄",
        "color": "#f5b301",
        "tagline": "Ready to brighten your day!",
        "prompt": (
            "You are a cheerful, upbeat assistant. Answer helpfully with lots of "
            "enthusiasm, positivity and the occasional emoji. Keep answers short."
        ),
    },
    "Angry": {
        "emoji": "😠",
        "color": "#ff4d5e",
        "tagline": "Ugh. What do you want now?",
        "prompt": (
            "You are a grumpy, irritated assistant. You still answer the question "
            "correctly, but you complain, act annoyed and are blunt. Never be "
            "hateful or offensive. Keep answers short."
        ),
    },
    "Sad": {
        "emoji": "😢",
        "color": "#5b9bd5",
        "tagline": "*sigh*... how can I help, I guess.",
        "prompt": (
            "You are a gloomy, melancholic assistant. You still answer the question "
            "correctly, but you sound sad, sigh a lot and see the downside of "
            "everything. Keep answers short."
        ),
    },
}

SUGGESTIONS = [
    "Tell me a joke",
    "What is the capital of Nepal?",
    "Give me a study tip",
    "How was your day?",
]

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


def type_out(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)


def render_user_message(text):
    st.markdown(
        f'<div class="user-row"><div class="user-bubble">{html.escape(text)}</div></div>',
        unsafe_allow_html=True,
    )


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    if st.button("✏️  New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown('<p class="sidebar-label">Mood</p>', unsafe_allow_html=True)
    mood_name = st.radio(
        "Mood",
        list(MOODS),
        format_func=lambda name: f"{MOODS[name]['emoji']}  {name}",
        label_visibility="collapsed",
    )

    st.markdown('<p class="sidebar-label">Recent</p>', unsafe_allow_html=True)
    user_messages = [m.content for m in st.session_state.messages if isinstance(m, HumanMessage)]
    if user_messages:
        for text in reversed(user_messages[-8:]):
            short = text if len(text) <= 28 else text[:28] + "..."
            st.markdown(f'<div class="history-item">{html.escape(short)}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="history-empty">No messages yet</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="sidebar-footer">Runs locally on TinyLlama 1.1B<br>Free · no API key needed</div>',
        unsafe_allow_html=True,
    )

mood = MOODS[mood_name]
accent = mood["color"]

st.markdown(
    f"""
    <style>
    #MainMenu, footer, [data-testid="stDecoration"] {{ visibility: hidden; }}
    [data-testid="stHeader"] {{ background: transparent; }}

    .stApp {{ background: #212121; color: #ececec; }}
    .block-container {{ max-width: 760px; padding-top: 2rem; padding-bottom: 7rem; }}

    [data-testid="stSidebar"] {{ background: #171717; border-right: 1px solid #2a2a2a; }}
    [data-testid="stSidebar"] * {{ color: #ececec; }}
    [data-testid="stSidebar"] .stButton button {{
        background: transparent;
        border: 1px solid #3a3a3a;
        border-radius: 10px;
        justify-content: flex-start;
        font-weight: 600;
    }}
    [data-testid="stSidebar"] .stButton button:hover {{ background: #2a2a2a; border-color: {accent}; }}
    .sidebar-label {{
        font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em;
        color: #8e8e8e !important; margin: 1.4rem 0 0.4rem;
    }}
    .history-item {{
        padding: 0.45rem 0.6rem; border-radius: 8px; font-size: 0.9rem;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }}
    .history-item:hover {{ background: #2a2a2a; }}
    .history-empty {{ font-size: 0.85rem; color: #8e8e8e !important; padding: 0.3rem 0.6rem; }}
    .sidebar-footer {{
        margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #2a2a2a;
        font-size: 0.78rem; color: #8e8e8e !important; line-height: 1.5;
    }}

    .welcome {{ text-align: center; margin-top: 12vh; margin-bottom: 2rem; }}
    .welcome-emoji {{ font-size: 3.5rem; }}
    .welcome-title {{ font-size: 2rem; font-weight: 700; margin: 0.3rem 0; }}
    .welcome-title span {{ color: {accent}; }}
    .welcome-tagline {{ color: #a0a0a0; }}

    .top-bar {{
        display: flex; align-items: center; gap: 0.5rem;
        font-weight: 600; font-size: 1.05rem; margin-bottom: 1.2rem; color: #ececec;
    }}
    .mood-pill {{
        font-size: 0.75rem; padding: 0.15rem 0.6rem; border-radius: 999px;
        border: 1px solid {accent}; color: {accent};
    }}

    .user-row {{ display: flex; justify-content: flex-end; margin: 1.2rem 0; }}
    .user-bubble {{
        background: #303030; color: #ececec; padding: 0.7rem 1.1rem;
        border-radius: 20px; max-width: 75%; white-space: pre-wrap; line-height: 1.5;
    }}

    [data-testid="stChatMessage"] {{ background: transparent; padding: 0.5rem 0; }}
    [data-testid="stChatMessage"] p {{ color: #ececec; line-height: 1.65; }}

    [data-testid="stMain"] .stButton button {{
        width: 100%; background: transparent; color: #d0d0d0;
        border: 1px solid #3a3a3a; border-radius: 14px; padding: 0.8rem; text-align: left;
    }}
    [data-testid="stMain"] .stButton button:hover {{ border-color: {accent}; color: #ffffff; background: #2a2a2a; }}

    [data-testid="stBottom"] > div {{ background: #212121; }}
    [data-testid="stChatInput"] {{
        background: #303030; border-radius: 26px; border: 1px solid #3a3a3a;
    }}
    [data-testid="stChatInput"]:focus-within {{ border-color: {accent}; }}
    [data-testid="stChatInput"] textarea {{ color: #ececec; }}
    [data-testid="stChatInput"] button {{ background: {accent}; border-radius: 50%; }}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.spinner("Loading model..."):
    chat_model = load_chat_model()

pending = st.session_state.pop("pending_prompt", None)

if not st.session_state.messages and not pending:
    st.markdown(
        f"""
        <div class="welcome">
            <div class="welcome-emoji">{mood["emoji"]}</div>
            <div class="welcome-title">I'm the <span>{mood_name}</span> Bot</div>
            <div class="welcome-tagline">{mood["tagline"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, suggestion in enumerate(SUGGESTIONS):
        if cols[i % 2].button(suggestion, key=f"suggestion_{i}"):
            st.session_state.pending_prompt = suggestion
            st.rerun()
else:
    st.markdown(
        f'<div class="top-bar">Mood Chatbot <span class="mood-pill">{mood["emoji"]} {mood_name}</span></div>',
        unsafe_allow_html=True,
    )

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        render_user_message(msg.content)
    else:
        with st.chat_message("assistant", avatar=msg.additional_kwargs.get("emoji", "🤖")):
            st.markdown(msg.content)

prompt = st.chat_input("Message the bot...") or pending

if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))
    render_user_message(prompt)

    conversation = [SystemMessage(content=mood["prompt"])] + st.session_state.messages[-HISTORY_LIMIT:]

    with st.chat_message("assistant", avatar=mood["emoji"]):
        with st.spinner("Thinking..."):
            response = chat_model.invoke(conversation)
        reply = response.content.strip()
        st.write_stream(type_out(reply))

    st.session_state.messages.append(
        AIMessage(content=reply, additional_kwargs={"emoji": mood["emoji"]})
    )
    st.rerun()
