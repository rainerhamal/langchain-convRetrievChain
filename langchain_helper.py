import openai
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader # type: ignore
from langchain_community.vectorstores import FAISS # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage

from dotenv import load_dotenv

load_dotenv()

# !Initialise the model
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# !Conversation Retrieval Chain
# Create chain that will take in the most recent input (input) and the conversation history (chat_history) and use an LLM to generate a search query.
# !Guide LLM response with a prompt template.Prompt templates convert raw user input to better input to the LLM.
# First we need a prompt that we can pass into an LLM to generate this search query
# prompt = ChatPromptTemplate.from_messages([
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("user", "{input}"),
#     ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
# ])

# loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
# docs = loader.load()
# embeddings = OpenAIEmbeddings()
# text_splitter = RecursiveCharacterTextSplitter()
# documents = text_splitter.split_documents(docs)
# vector = FAISS.from_documents(documents, embeddings)
# retriever = vector.as_retriever()
# retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

# test this out by passing in an instance where the user asks a follow-up question.
# Assuming the chat_history content is ias stated below
# chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
# response = retriever_chain.invoke({
#     "chat_history": chat_history,
#     "input": "Tell me how"
# })
# print(response)

# ! create a new chain to continue the conversation with these retrieved documents in mind
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()
embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()

prompt1 = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
])
retriever_chain = create_history_aware_retriever(llm, retriever, prompt1)

prompt2 = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the provided context: \n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])
document_chain = create_stuff_documents_chain(llm, prompt2)
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
response = retrieval_chain.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
})
print(response)