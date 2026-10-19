## local model free to use

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
     task="text-generation",
      pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    ),
)
chat_model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content="You are a helpful assistant that can answer questions and help with tasks."),
]
print ("-------Welcome to the chatbot type 0 to exit the application-------")
while True:
      
      prompt = input("You:")
      messages.append(HumanMessage(content=prompt))
      if prompt == "0":
        break
      response = chat_model.invoke(prompt)
      messages.append(AIMessage(content=response.content))
      print(f"Assistant: {response.content}")
print(messages)