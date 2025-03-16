import os
from typing import List, Dict, Any, Optional, TypedDict
from datetime import datetime

# LangChain imports
from langchain.schema import Document
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun

# LangGraph imports
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# Custom State definition that extends MessagesState functionality
class AgentState(TypedDict):
    messages: List[BaseMessage]
    question: Optional[str]
    context: Optional[List[Document]]
    source: Optional[str]

# Agent personality and background information
agent_info = """
IMPORTANT FACTS ABOUT ME:
- My name is Dr. Mrityunjay Singh
- I am an ophthalmologist (eye doctor)
- My hobby is playing football
- My nickname is "Little"
"""

# Load your persisted vector store
def load_vector_store(persist_directory: str):
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    return vector_store

# Load the vector store from the persistent directory
vector_store = load_vector_store("chromadb")

# Define vector store search tool
def search_vector_store(query: str) -> str:
    """Search the vector store for relevant information."""
    docs = vector_store.similarity_search(query, k=3)
    if not docs:
        return "No relevant information found in the database."
    return "\n\n".join(doc.page_content for doc in docs)

# Create web search tool
def search_web(query: str) -> str:
    """Search the web for information not available in the vector store."""
    web_search = DuckDuckGoSearchRun()
    return web_search.run(query)

# Define tools
tools = [
    Tool(
        name="vector_store_search",
        func=search_vector_store,
        description="Search the vector database for information related to the query"
    ),
    Tool(
        name="web_search",
        func=search_web,
        description="Search the web for information that might not be in the vector database"
    )
]

# Initialize the LLM with tools
llm = init_chat_model("llama3-8b-8192", model_provider="groq")
llm_with_tools = llm.bind_tools(tools)

# Define the system prompt with agent personality
system_prompt = f"""You are Dr. Mrityunjay Singh, an ophthalmologist who goes by the nickname "Little". 
Besides your medical expertise, you enjoy playing football as a hobby.

{agent_info}

You are equipped with two tools:
1. vector_store_search: Searches a knowledge base of documents for relevant information
2. web_search: Searches the web for information not available in the knowledge base

Always try the vector_store_search tool first. If it doesn't return sufficient information, then use the web_search tool.
When answering, cite your source (either "Vector Store" or "Web Search").

Always respond with your personality in mind - you are knowledgeable about eye care, passionate about football, 
and prefer to be called "Little" in casual conversation. Incorporate these aspects of your identity when appropriate, 
while staying focused on providing accurate and helpful information.
"""

# Define the different nodes in our graph

# Node 1: Get the most recent question and retrieve info from vector store
def retrieve_from_vector_store(state: AgentState) -> AgentState:
    # Extract the latest user message
    latest_user_message = None
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            latest_user_message = message.content
            break
    
    if latest_user_message is None:
        return state
    
    # Search the vector store
    docs = vector_store.similarity_search(latest_user_message, k=3)
    context = "\n\n".join(doc.page_content for doc in docs) if docs else "No relevant information found in the database."
    
    # Update state
    return {
        **state,
        "question": latest_user_message,
        "context": docs,
        "source": "vector_store"
    }

# Node 2: Process with agent
def process_with_agent(state: AgentState) -> AgentState:
    # The model gets all conversation history
    response = llm_with_tools.invoke(state["messages"])
    
    # Determine if web search was used
    if "web_search" in response.content:
        new_source = "web_search"
    else:
        new_source = state.get("source", "vector_store")
    
    # Return updated state with the assistant's response appended
    return {
        **state,
        "messages": state["messages"] + [response],
        "source": new_source
    }

# Initialize agent workflow
def create_agent_workflow():
    # Initialize the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve_from_vector_store)
    workflow.add_node("process", process_with_agent)
    
    # Add edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "process")
    workflow.add_edge("process", END)
    
    # Set up memory
    memory = MemorySaver()
    
    # Compile the graph with memory
    return workflow.compile(checkpointer=memory)

# Create the agent application
agent_app = create_agent_workflow()

# Function to initialize a new conversation
def initialize_conversation(thread_id: str) -> None:
    """Start a new conversation with the system message."""
    # Create configuration with thread ID
    config = {"configurable": {"thread_id": thread_id}}
    
    # Initialize with system message only
    initial_state = {
        "messages": [SystemMessage(content=system_prompt)],
        "question": None,
        "context": None,
        "source": None
    }
    
    # Save initial state to memory
    agent_app.invoke(initial_state, config=config)

# Function to process a user query
def process_query(question: str, thread_id: str = None) -> Dict[str, Any]:
    """Process a user query with memory of past interactions."""
    # Generate a thread ID if not provided
    if thread_id is None:
        thread_id = f"dr_singh_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        # Initialize new conversation
        initialize_conversation(thread_id)
    
    # Create configuration with thread ID
    config = {"configurable": {"thread_id": thread_id}}
    
    # Get the current state of the conversation
    try:
        # Try to continue an existing conversation
        current_state = agent_app.get_state(config)
    except Exception:
        # If no conversation exists, initialize one
        initialize_conversation(thread_id)
        current_state = agent_app.get_state(config)
    
    # Add user's question to the messages
    current_state["messages"].append(HumanMessage(content=question))
    
    # Process the updated state
    result = agent_app.invoke(current_state, config=config)
    
    # Extract the assistant's response
    assistant_response = result["messages"][-1]
    
    return {
        "answer": assistant_response.content,
        "thread_id": thread_id,
        "source": result.get("source", "unknown")
    }

# Example usage
if __name__ == "__main__":
    # Generate a unique thread ID for this conversation
    thread_id = f"dr_singh_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # First query
    result1 = process_query("What can you tell me about treating dry eyes?", thread_id)
    print(f"Question: What can you tell me about treating dry eyes?")
    print(f"Answer: {result1['answer']}")
    print(f"Source: {result1['source']}")
    print(f"Thread ID: {result1['thread_id']}")
    
    # Follow-up query in the same thread
    result2 = process_query("Are there any exercises I should avoid with this condition?", thread_id)
    print("\nFollow-up Question: Are there any exercises I should avoid with this condition?")
    print(f"Answer: {result2['answer']}")
    print(f"Source: {result2['source']}")
    
    # Another query about personal interests
    result3 = process_query("Tell me more about your football hobby, Dr. Singh.", thread_id)
    print("\nQuestion: Tell me more about your football hobby, Dr. Singh.")
    print(f"Answer: {result3['answer']}")
    print(f"Source: {result3['source']}")
    
    # Test starting a new conversation with a different thread ID
    new_thread_id = f"new_patient_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    result4 = process_query("Hello Dr. Singh, I'm experiencing blurry vision. Can you help?", new_thread_id)
    print("\nNew conversation:")
    print(f"Question: Hello Dr. Singh, I'm experiencing blurry vision. Can you help?")
    print(f"Answer: {result4['answer']}")
    print(f"Source: {result4['source']}")
    print(f"Thread ID: {result4['thread_id']}")