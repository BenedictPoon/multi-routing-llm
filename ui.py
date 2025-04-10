import streamlit as st
from router import process_query

st.set_page_config(
    page_title="MultiLLM Query Router",
    page_icon="🤖",
    layout="wide"
)

# Add styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        color: #333333;
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
        color: #333333 !important;
    }
    .stMarkdown, .stText {
        color: #333333 !important;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown("<h1 class='title'>MultiLLM Query Router</h1>", unsafe_allow_html=True)
st.markdown("""
This application classifies and routes your query to the appropriate handler based on the content.
Simply enter your question or request below and our AI will provide a relevant response.
""")

# User input
query = st.text_area("Enter your question or request:", height=100)
submit_button = st.button("Submit")

# Initialize chat history in session state if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Process the query
if submit_button and query:
    with st.spinner("Processing your query..."):
        # Call the process_query function from router.py
        result = process_query(query)
        
        # Add to chat history
        st.session_state.chat_history.append({
            "query": query,
            "category": result["category"],
            "response": result["response"]
        })

# Display chat history
if st.session_state.chat_history:
    st.subheader("Conversation History")
    
    for i, interaction in enumerate(st.session_state.chat_history):
        # Display user query in a chat bubble
        st.markdown(f"""
        <div style="background-color: #E8F4F8; border-radius: 10px; padding: 10px; margin: 5px 0; text-align: right;">
            <p style="margin: 0; color: #333333;"><strong>You:</strong> {interaction["query"]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display the category
        category = interaction["category"]
        category_colors = {
            "technical_question": "#1E88E5",  # Blue
            "product_inquiry": "#43A047",     # Green
            "customer_support": "#E53935",    # Red
            "general_inquiry": "#FB8C00",     # Orange
            "other": "#8E24AA"                # Purple
        }
        color = category_colors.get(category, "#757575")  # Default to gray
        
        # Display assistant response with category tag
        st.markdown(f"""
        <div style="background-color: #F8F8F8; border-radius: 10px; padding: 10px; margin: 5px 0;">
            <span style="background-color: {color}; color: white; padding: 3px 8px; border-radius: 10px; font-size: 0.7rem; margin-bottom: 5px; display: inline-block;">
                {category.replace('_', ' ').title()}
            </span>
            <p style="margin: 5px 0 0 0; color: #333333;"><strong>Assistant:</strong> {interaction["response"]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add a divider if it's not the last item
        if i < len(st.session_state.chat_history) - 1:
            st.markdown("<hr style='margin: 15px 0; opacity: 0.3;'>", unsafe_allow_html=True)

# Add sidebar with information
with st.sidebar:
    st.title("MultiLLM Router")
    st.markdown("""
    This application demonstrates how to use:
    - **LangGraph** for creating an AI workflow with routing
    - **LangChain** for working with LLMs
    - **Streamlit** for building the user interface
    
    It classifies queries into these categories:
    - Technical Questions
    - Product Inquiries
    - Customer Support
    - General Inquiries
    - Other
    """)
    
    # Add debug tools in sidebar
    st.subheader("Debug Tools")
    if st.button("Clear Conversation"):
        st.session_state.chat_history = []
        st.experimental_rerun()
    
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