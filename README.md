# Basic ChatBot

A beginner Generative AI project built with [LangChain](https://python.langchain.com/). It includes a free chatbot that runs locally with a Streamlit web interface, plus examples of using hosted chat models and embedding models.

**Live demo:** [aashrawat-basic-chatbot-chatmodelsapp-9uexc7.streamlit.app](https://aashrawat-basic-chatbot-chatmodelsapp-9uexc7.streamlit.app/)

## Project structure

```
chatmodels/
  app.py              Streamlit chatbot web app (local TinyLlama model, free)
  localmodel.py       Same chatbot in the terminal
  huggingface.py      Chat model via the Hugging Face Inference API
  chat.py             Chat model via OpenAI (paid)
embeddingsmodel/
  embeddings.py       OpenAI text embeddings (paid)
  hugging_faceEmb.py  Local sentence-transformers embeddings (free)
requirements.txt
```

## Setup

1. Install [uv](https://docs.astral.sh/uv/) and create a virtual environment:

   ```powershell
   uv venv
   .venv\Scripts\Activate.ps1
   ```

2. Install the dependencies:

   ```powershell
   uv pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with the API keys you need (only required for the hosted-model scripts):

   ```
   OPENAI_API_KEY="your-openai-key"
   GROQ_API_KEY="your-groq-key"
   GOOGLE_API_KEY="your-google-key"
   HUGGINGFACEHUB_ACCESS_TOKEN="your-huggingface-token"
   ```

## Run the chatbot

```powershell
cd chatmodels
streamlit run app.py
```

Then open http://localhost:8501 in your browser. The first run downloads the TinyLlama model (about 2 GB). Type `0` to end the chat.

To chat in the terminal instead:

```powershell
cd chatmodels
py localmodel.py
```

## Run the embedding examples

```powershell
cd embeddingsmodel
py hugging_faceEmb.py
```
