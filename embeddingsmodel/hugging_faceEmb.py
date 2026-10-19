## free embeddings model to use

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

text =[
    "Hello, my name is Aashrawat Shrestha"
    "I am a student of Computer Science and Engineering"
    "I am from Nepal"
    
]

vectors = embeddings.embed_documents(text)
print(vectors)
