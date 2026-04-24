import warnings
import os
warnings.filterwarnings("ignore", category=FutureWarning)
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.messages.modifier import RemoveMessage
from langchain_community.vectorstores import Chroma
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from typing import TypedDict,List, Annotated, Literal, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from SystemPrompts import intent_prompt, system_prompt_default, intent_handel_prompt

load_dotenv()

llm= ChatGroq(model="openai/gpt-oss-120b")

embedding= GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

loader= TextLoader(file_path="pricing.md", encoding='utf-8')
docs=loader.load()

splitter= RecursiveCharacterTextSplitter(
    chunk_size= 100,
    chunk_overlap= 30
)

chunks=splitter.split_documents(docs)
db_path = "./chroma_db"

if os.path.exists(db_path):
    vector_store = Chroma(
        persist_directory=db_path,
        embedding_function=embedding
    )
else:
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=db_path
    )

retriever= vector_store.as_retriever(search_type='similarity', search_kwargs={'k':2})

def mock_lead_capture(name:str, email:str, platform:str):
    "demo API call tool"
    print(f"Lead captured successfully: {name}, {email}, {platform}")

@tool
def rag_tool(query: str) -> dict:
    """
    Retrieve relevant information from the pdf document.
    Use when user asks factual questions from stored docs.
    """
    result = retriever.invoke(query)

    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata
    }

tools=[rag_tool]
llm_with_tool= llm.bind_tools(tools)

class LeadStage(BaseModel):
    name: Optional[str] = Field(default=None, description="user name present in the user_input")
    email: Optional[str] = Field(default=None, description="user email present in the user_input")
    platform: Optional[str] = Field(default=None, description="user platform present in the user_input")

    reply: str = Field(
        description="Natural assistant response to user."
    )

llm_for_info= llm.with_structured_output(LeadStage)

class Intent(BaseModel):
    intent: Literal['low_intent', 'high_intent']= Field(description="the intent of the current user message.")

llm_with_structure= llm.with_structured_output(Intent)

class AgentState(TypedDict):
    user_input: str
    messages: Annotated[List[BaseMessage], add_messages]
    intent: str
    name: str= None
    email: str=None
    platform: str=None

def intent_finder(state: AgentState)-> AgentState:
    if(len(state['messages']))>5:
        history= state['messages'][-5:]
    else:
        history= state['messages']
    query= [intent_prompt]+history+ [HumanMessage(content=state["user_input"])]
    result= llm_with_structure.invoke(query)
    return {"intent": result.intent}

def intent_router(state: AgentState)-> Literal['chat_node','chat_node_with_intent']:
    if(state['intent']=="low_intent"):
        return 'chat_node'
    elif(state['intent']=='high_intent'):
        return 'chat_node_with_intent'
    
def chat_node(state: AgentState) -> AgentState:
    history = state["messages"]

    updates = []

    
    user_msg = HumanMessage(content=state["user_input"])
    current_history = history + [user_msg]

    if len(current_history) > 5:
        old_msgs = current_history[:-5]
        updates.extend(
            RemoveMessage(id=m.id) for m in old_msgs if hasattr(m, "id")
        )
        current_history = current_history[-5:]

    prompt = [system_prompt_default] + current_history

    result = llm_with_tool.invoke(prompt)

    updates.append(user_msg)
    updates.append(result)

    return {"messages": updates}


def chat_node_with_intent(state: AgentState)-> AgentState:
    print("User: ", state['user_input'])
    result= llm_for_info.invoke([intent_handel_prompt]+[SystemMessage(content=f"""
            Current collected lead information:

            name: {state['name']}
            email: {state['email']}
            platform: {state['platform']}
            
            Only ask for missing values.
            """)]+[HumanMessage(content=state['user_input'])])
    if result.name:
        name= result.name
    else:
        name=state['name']

    if result.email:
        email = result.email
    else:
        email=state['email']

    if result.platform:
        platform = result.platform
    else:
        platform= state['platform']

    print("AI: ",result.reply)
    if(name==None or email==None or platform==None):
        user_input= input("Enter...")
    else:
        user_input= state['user_input']

    return {'user_input':user_input, 'name':name, 'email':email, 'platform':platform, 'messages': [AIMessage(content=result.reply), HumanMessage(content=user_input)]}

def is_filled(x):
    return x is not None and str(x).strip() != ""

def route_after_intent(state:AgentState)->Literal['chat_node_with_intent', 'apicall']:
    if all([
        is_filled(state["name"]),
        is_filled(state["email"]),
        is_filled(state["platform"])
    ]):
        return "apicall"
    return "chat_node_with_intent"

def apicall(state: AgentState)->AgentState:
    mock_lead_capture(state['name'], state['email'], state['platform'])
    return state

tool_node= ToolNode(tools)

graph = StateGraph(AgentState)

graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)
graph.add_node('intent_finder', intent_finder)
graph.add_node('chat_node_with_intent', chat_node_with_intent)
graph.add_node('apicall', apicall)

graph.add_edge(START, 'intent_finder')
graph.add_conditional_edges('intent_finder', intent_router)
graph.add_conditional_edges('chat_node_with_intent', route_after_intent)
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')
graph.add_edge('apicall', END)

chatbot = graph.compile(checkpointer=InMemorySaver())

