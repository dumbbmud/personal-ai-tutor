import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage

load_dotenv()

repo_id = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
base_llm = HuggingFaceEndpoint(
    repo_id = repo_id,
    temperature = 0.7,
    max_new_tokens = 512,
)

llm = ChatHuggingFace(llm=base_llm)

print("Hello, I am your personalized AI Tutor (powered by Open Source). Ask me any question, or type 'exit' to quit.")

while True:
    user_input = input('You: ')
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break 
    
    messages = [HumanMessage(content=user_input)]

    response = llm.invoke(messages) 

    print(f"AI Tutor: {response.content}")
      