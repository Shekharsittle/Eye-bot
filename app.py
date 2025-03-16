import streamlit as st
import uuid
from datetime import datetime
import os

# Import the agent functions from chat.py
from chat import process_query, initialize_conversation

# Page configuration
st.set_page_config(
    page_title="Dr. Mrityunjay Singh - AI Ophthalmologist",
    page_icon="👁️",
    layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem; 
        border-radius: 0.5rem; 
        margin-bottom: 1rem; 
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #ECEFF1;
        border-left: 5px solid #90CAF9;
    }
    .chat-message.assistant {
        background-color: #E3F2FD;
        border-left: 5px solid #42A5F5;
    }
    .chat-message .source {
        font-size: 0.8rem;
        color: #546E7A;
        margin-top: 0.5rem;
        align-self: flex-end;
    }
    .chat-message .timestamp {
        font-size: 0.7rem;
        color: #78909C;
        margin-top: 0.25rem;
        align-self: flex-end;
    }
    .stTextInput {
        position: fixed;
        bottom: 3rem;
        width: calc(100% - 2rem);
    }
    .chat-container {
        margin-bottom: 5rem;
    }
    .sidebar-content {
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    # Generate a unique thread ID for this session
    st.session_state.thread_id = f"streamlit_{str(uuid.uuid4())}"
    # Initialize the conversation
    initialize_conversation(st.session_state.thread_id)

# Sidebar with information about Dr. Singh
with st.sidebar:
    st.image("https://img.freepik.com/free-photo/doctor-with-his-arms-crossed-white-background_1368-5790.jpg", width=200)
    st.title("Dr. Mrityunjay Singh")
    st.subheader("Ophthalmologist")
    
    st.markdown("""
    <div class="sidebar-content">
        <p><strong>Nickname:</strong> Little</p>
        <p><strong>Specialty:</strong> Eye Care & Treatment</p>
        <p><strong>Hobby:</strong> Football</p>
        <hr>
        <p>Dr. Singh can answer your questions about eye health and treatments. 
        He uses a knowledge base for medical information and can search the web when needed.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Option to start a new conversation
    if st.button("Start New Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = f"streamlit_{str(uuid.uuid4())}"
        initialize_conversation(st.session_state.thread_id)
        st.success("New conversation started!")

# Main title
st.title("Consult with Dr. Mrityunjay Singh")
st.markdown("Ask me anything about eye health, treatments, or eye conditions.")

# Display chat messages
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="chat-message user">
            <div>{message["content"]}</div>
            <div class="timestamp">{message["timestamp"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant">
            <div>{message["content"]}</div>
            <div class="source">Source: {message["source"]}</div>
            <div class="timestamp">{message["timestamp"]}</div>
        </div>
        """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Chat input
with st.container():
    user_input = st.text_input("Your question:", key="user_input", placeholder="Type your eye health question here...")

    if user_input:
        # Add user message to chat history
        timestamp = datetime.now().strftime("%I:%M %p, %b %d")
        st.session_state.messages.append({
            "role": "user", 
            "content": user_input,
            "timestamp": timestamp
        })
        
        # Get response from Dr. Singh
        result = process_query(user_input, st.session_state.thread_id)
        
        # Add assistant's response to chat history
        timestamp = datetime.now().strftime("%I:%M %p, %b %d")
        st.session_state.messages.append({
            "role": "assistant", 
            "content": result["answer"],
            "source": result["source"],
            "timestamp": timestamp
        })
        
        # Rerun to update UI
        st.experimental_rerun()