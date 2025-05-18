import os
from typing import Annotated, Literal, TypedDict, Union
import requests
import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
# Import from the classifier
from huggingface_classifier import classify_input
# Import the sensitivity checker
from sensitivity_classifier.masking import check_sensitivity

import langgraph as lg
from langgraph.graph import END, StateGraph

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Define our state
class State(TypedDict):
    messages: list
    category: str
    response: str
    use_ollama: bool

# Define categories
CATEGORIES = ["technical_question", "general_inquiry"]

# Function to call OpenAI API directly
def call_openai_api(messages, model="gpt-4"):
    logger.info(f"Calling OpenAI API with model: {model}")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key not set")
        return "API key not set"
        
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            data=json.dumps(data)
        )
        response.raise_for_status()
        logger.info("OpenAI API call successful")
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        return f"Error: {str(e)}"

# Create the classifier node
def classifier(state: State) -> dict:
    """Classify the incoming user message"""
    logger.info("Running classifier")
    # Get the last user message
    last_message = state["messages"][-1]
    user_input = last_message.content
    logger.info(f"Classifying input: {user_input}")
    
    # Use the rule-based classifier
    category = classify_input(user_input)
    logger.info(f"Classification result: {category}")
    
    # If category is not technical or general, default to general
    if category not in CATEGORIES:
        logger.info(f"Category {category} not in allowed categories, defaulting to general_inquiry")
        category = "general_inquiry"
    
    return {
        "category": category,
        "messages": [HumanMessage(content=user_input)]
    }

# Define router to determine next node
def router(state: State) -> str:
    """Route to the appropriate handler based on the category"""
    category = state["category"]
    logger.info(f"Routing to handler for category: {category}")
    
    if category == "technical_question":
        return "technical_handler"
    else:
        return "general_handler"

# Define handlers for each category
def technical_handler(state: State) -> dict:
    """Handle technical questions"""
    logger.info("Running technical handler")
    last_message = state["messages"][-1]
    user_input = last_message.content
    use_ollama = state.get("use_ollama", False)
    logger.info(f"Using {'Ollama' if use_ollama else 'OpenAI'} for technical question")
    
    messages = [{"role": "user", "content": user_input}]
    if use_ollama:
        response = call_ollama_api(messages, model=OLLAMA_MODEL)
    else:
        response = call_openai_api(messages, model="gpt-4")
    return {"response": response}

def general_handler(state: State) -> dict:
    """Handle general inquiries"""
    logger.info("Running general handler")
    last_message = state["messages"][-1]
    user_input = last_message.content
    use_ollama = state.get("use_ollama", False)
    logger.info(f"Using {'Ollama' if use_ollama else 'OpenAI'} for general inquiry")
    
    messages = [{"role": "user", "content": user_input}]
    if use_ollama:
        response = call_ollama_api(messages, model=OLLAMA_MODEL)
    else:
        response = call_openai_api(messages, model="gpt-4")
    return {"response": response}

def call_ollama_api(messages, model="llama3"):
    """
    Call the local Ollama server for inference.
    messages: list of dicts with 'role' and 'content'
    model: the Ollama model to use (e.g., 'llama3', 'mistral', etc.)
    """
    logger.info(f"Calling Ollama API with model: {model}")
    try:
        # Ollama expects a single prompt string, so concatenate messages
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(
            "http://localhost:11434/api/generate",
            headers={"Content-Type": "application/json"},
            data=json.dumps(data)
        )
        response.raise_for_status()
        logger.info("Ollama API call successful")
        return response.json()["response"]
    except Exception as e:
        logger.error(f"Ollama API error: {str(e)}")
        return f"Ollama error: {str(e)}"

# Build the graph
def build_graph():
    logger.info("Building workflow graph")
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("classifier", classifier)
    workflow.add_node("technical_handler", technical_handler)
    workflow.add_node("general_handler", general_handler)
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "classifier",
        router,
        {
            "technical_handler": "technical_handler",
            "general_handler": "general_handler"
        }
    )
    
    # Add edges from handlers to END
    workflow.add_edge("technical_handler", END)
    workflow.add_edge("general_handler", END)
    
    # Set the entry point
    workflow.set_entry_point("classifier")
    
    logger.info("Workflow graph built successfully")
    return workflow.compile()

# Create the graph
graph = build_graph()

# Environment variables
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3:latest")
logger.info(f"Using Ollama model: {OLLAMA_MODEL}")

def process_query(query, use_ollama=None):
    """
    Process a user query through the graph.
    If use_ollama is None, automatically select based on sensitivity.
    """
    logger.info(f"Processing query: {query}")
    logger.info(f"Initial use_ollama parameter: {use_ollama}")
    
    # If use_ollama is None, check sensitivity to determine model
    if use_ollama is None:
        logger.info("Checking sensitivity for automatic model selection")
        is_sensitive, masked_text = check_sensitivity(query)
        use_ollama = is_sensitive
        logger.info(f"Sensitivity check result: {'sensitive' if is_sensitive else 'not sensitive'}")
        if is_sensitive:
            logger.info(f"Masked text: {masked_text}")
        logger.info(f"Selected model: {'local (Ollama)' if use_ollama else 'cloud (OpenAI)'}")
    
    # Initialize state
    state = {
        "messages": [HumanMessage(content=query)],
        "category": "",
        "response": "",
        "use_ollama": use_ollama
    }
    logger.info(f"Initialized state with use_ollama={use_ollama}")
    
    # Execute the graph
    try:
        logger.info("Executing workflow graph")
        result = graph.invoke(state)
        logger.info("Workflow execution completed successfully")
        
        # Return the final response
        return {
            "category": result["category"],
            "response": result["response"],
            "original_query": query,
            "model_used": "ollama" if state["use_ollama"] else "openai"
        }
    except Exception as e:
        logger.error(f"Error in workflow execution: {str(e)}")
        return {
            "category": "general_inquiry",
            "response": f"Error processing query: {str(e)}",
            "original_query": query,
            "model_used": "ollama" if state["use_ollama"] else "openai"
        }

# Test the routing system
if __name__ == "__main__":
    logger.info("Starting test sequence")
    test_inputs = [
        "How do I install your Python library?",  # Technical
        "My email is john@example.com",  # General + Sensitive
        "What's the price of your enterprise plan?",  # General
        "How do I configure the API?",  # Technical
        "Tell me about your company"  # General
    ]
    
    for input_text in test_inputs:
        logger.info(f"\nTesting input: {input_text}")
        result = process_query(input_text)  # Let it auto-select the model
        logger.info(f"Category: {result['category']}")
        logger.info(f"Model Used: {result['model_used']}")
        logger.info(f"Response: {result['response']}")
        logger.info("-" * 70)