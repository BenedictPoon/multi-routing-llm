import streamlit as st
from router import process_query

st.set_page_config(
    page_title="MultiLLM Query Router",
    page_icon="🤖",
    layout="wide"
)

# Initialize chat history in session state if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Update CSS to make headings and text white
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        color: white !important;
    }
    .response-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    .category-tag {
        background-color: #4CAF50;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.8rem;
        margin-right: 10px;
    }
    p, div, span, h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }
    .stMarkdown, .stText {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Display chat history at the top
if st.session_state.chat_history:
    st.subheader("Conversation History")
    
    # Create a scrollable container for the chat history
    chat_container = st.container()
    with chat_container:
        for i, interaction in enumerate(st.session_state.chat_history):
            # Display user query in a chat bubble
            st.markdown(f"""
            <div style="background-color: #2c3e50; border-radius: 10px; padding: 10px; margin: 5px 0; text-align: right;">
                <p style="margin: 0; color: white;"><strong>You:</strong> {interaction["query"]}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display the category
            category = interaction["category"]
            category_colors = {
                "technical_question": "#1E88E5",  # Blue
                "general_inquiry": "#FB8C00",     # Orange
            }
            color = category_colors.get(category, "#757575")  # Default to gray
            
            # Display assistant response with category tag
            st.markdown(f"""
            <div style="background-color: #34495e; border-radius: 10px; padding: 10px; margin: 5px 0;">
                <span style="background-color: {color}; color: white; padding: 3px 8px; border-radius: 10px; font-size: 0.7rem; margin-bottom: 5px; display: inline-block;">
                    {category.replace('_', ' ').title()}
                </span>
                <p style="margin: 5px 0 0 0; color: white;"><strong>Assistant:</strong> {interaction["response"]}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Add a divider if it's not the last item
            if i < len(st.session_state.chat_history) - 1:
                st.markdown("<hr style='margin: 15px 0; opacity: 0.3;'>", unsafe_allow_html=True)
        
        # Auto-scroll to the bottom
        st.markdown("<script>window.scrollTo(0, document.body.scrollHeight);</script>", unsafe_allow_html=True)

# Header at the bottom
st.markdown("<h1 class='title'>MultiLLM Query Router</h1>", unsafe_allow_html=True)

# Description text just above the prompt
st.markdown("""
This application classifies and routes your query to the appropriate handler based on the content.
Simply enter your question or request below and our AI will provide a relevant response.
""")

# User input at the bottom
st.markdown("<h2>Ask a Question</h2>", unsafe_allow_html=True)
query = st.text_area("Enter your question or request:", height=100)

# Model selection as a segmented button
model_choice = st.radio("Choose Model", ["Auto", "Cloud", "Local"], horizontal=True)

# Determine which model to use based on selection
if model_choice == "Cloud":
    st.session_state["use_ollama"] = False
    st.session_state["model_choice"] = "cloud"
elif model_choice == "Local":
    st.session_state["use_ollama"] = True
    st.session_state["model_choice"] = "local"
elif model_choice == "Auto":
    st.session_state["model_choice"] = "auto"

# Submit button
submit_button = st.button("Submit")

# Process the query
if submit_button and query:
    with st.spinner("Processing your query..."):
        # Determine which model to use based on selection
        if model_choice == "Cloud":
            use_ollama = False
        elif model_choice == "Local":
            use_ollama = True
        else:  # Auto
            use_ollama = None  # Let the router decide based on sensitivity
            
        # Call the process_query function from router.py
        result = process_query(query, use_ollama=use_ollama)
        
        # Add to chat history
        st.session_state.chat_history.append({
            "query": query,
            "category": result["category"],
            "response": result["response"]
        })
        
        # Clear the prompt window
        query = ""
        st.rerun()  # Rerun the app to update the UI

# Add sidebar with information
with st.sidebar:
    st.title("MultiLLM Router")
    st.markdown("""
    This application demonstrates how to use:
    - **LangGraph** for creating an AI workflow with routing
    - **LangChain** for working with LLMs
    - **Streamlit** for building the user interface
    
    It classifies queries into these categories:
    - Technical Questions: For technical support and implementation queries
    - General Inquiries: For all other questions and information requests
    """)
    
    # Add debug tools in sidebar
    st.subheader("Debug Tools")
    if st.button("Clear Conversation"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Show routing statistics if there's history
    if st.session_state.chat_history:
        st.subheader("Routing Statistics")
        
        # Count categories
        categories = {}
        for interaction in st.session_state.chat_history:
            cat = interaction["category"]
            if cat in categories:
                categories[cat] += 1
            else:
                categories[cat] = 1
        
        # Display as a simple bar chart
        for cat, count in categories.items():
            st.markdown(f"{cat.replace('_', ' ').title()}: {'▓' * count} ({count})")
    
    st.markdown("---")
    st.markdown("© 2025 MultiLLM Query Router")