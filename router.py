import os
from typing import Annotated, Literal, TypedDict, Union
from openai import OpenAI

from langchain_core.messages import HumanMessage, SystemMessage
# Import from the classifier
from huggingface_classifier import classify_input

import langgraph as lg
from langgraph.graph import END, StateGraph

# Initialize the OpenAI client using the API key from environment variables
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Define our state
class State(TypedDict):
    messages: list
    category: str
    response: str

# Define categories
CATEGORIES = ["technical_question", "product_inquiry", "customer_support", "general_inquiry", "other"]

# Create the classifier node
def classifier(state: State) -> dict:
    """Classify the incoming user message"""
    # Get the last user message
    last_message = state["messages"][-1]
    user_input = last_message.content
    
    # Use the rule-based classifier
    category = classify_input(user_input)
    
    # Update state with classification
    return {"category": category}

# Define router to determine next node
def router(state: State) -> str:
    """Route to the appropriate handler based on the category"""
    category = state["category"]
    
    if category == "technical_question":
        return "technical_handler"
    elif category == "product_inquiry":
        return "product_handler"
    elif category == "customer_support":
        return "support_handler"
    elif category == "general_inquiry":
        return "general_handler"
    else:
        return "other_handler"

# Define handlers for each category
def technical_handler(state: State) -> dict:
    """Handle technical questions"""
    # Get the last user message
    last_message = state["messages"][-1]
    user_input = last_message.content
    
    try:
        # Use OpenAI to generate a response
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            store=True,
            messages=[
                {"role": "system", "content": "You are a technical support specialist. Provide detailed, accurate and helpful responses to technical questions."},
                {"role": "user", "content": user_input}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        return {"response": f"Error using OpenAI API: {str(e)}"}

def product_handler(state: State) -> dict:
    """Handle product inquiries"""
    last_message = state["messages"][-1]
    user_input = last_message.content
    
    try:
        # Use OpenAI to generate a response
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            store=True,
            messages=[
                {"role": "system", "content": "You are a product specialist. Provide informative responses about product features, pricing, and comparisons."},
                {"role": "user", "content": user_input}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        return {"response": f"Error using OpenAI API: {str(e)}"}

def support_handler(state: State) -> dict:
    """Handle customer support issues"""
    last_message = state["messages"][-1]
    user_input = last_message.content
    
    try:
        # Use OpenAI to generate a response
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            store=True,
            messages=[
                {"role": "system", "content": "You are a customer support representative. Be empathetic and helpful when addressing customer issues."},
                {"role": "user", "content": user_input}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        return {"response": f"Error using OpenAI API: {str(e)}"}

def general_handler(state: State) -> dict:
    """Handle general inquiries"""
    last_message = state["messages"][-1]
    user_input = last_message.content
    
    try:
        # Use OpenAI to generate a response
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            store=True,
            messages=[
                {"role": "system", "content": "You are an information specialist. Provide clear and comprehensive information about the company, its services, and general questions."},
                {"role": "user", "content": user_input}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        return {"response": f"Error using OpenAI API: {str(e)}"}

def other_handler(state: State) -> dict:
    """Handle other types of queries"""
    last_message = state["messages"][-1]
    user_input = last_message.content
    
    try:
        # Use OpenAI to generate a response
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            store=True,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Provide friendly and informative responses to a wide range of queries."},
                {"role": "user", "content": user_input}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        return {"response": f"Error using OpenAI API: {str(e)}"}


# Build the graph
def build_graph():
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("classifier", classifier)
    workflow.add_node("technical_handler", technical_handler)
    workflow.add_node("product_handler", product_handler)
    workflow.add_node("support_handler", support_handler)
    workflow.add_node("general_handler", general_handler)
    workflow.add_node("other_handler", other_handler)
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "classifier",
        router,
        {
            "technical_handler": "technical_handler",
            "product_handler": "product_handler",
            "support_handler": "support_handler",
            "general_handler": "general_handler",
            "other_handler": "other_handler"
        }
    )
    
    # Add edges from handlers to END
    workflow.add_edge("technical_handler", END)
    workflow.add_edge("product_handler", END)
    workflow.add_edge("support_handler", END)
    workflow.add_edge("general_handler", END)
    workflow.add_edge("other_handler", END)
    
    # Set the entry point
    workflow.set_entry_point("classifier")
    
    return workflow.compile()

# Create the graph
graph = build_graph()

# Function to process a query through the graph
def process_query(query):
    """Process a user query through the graph"""
    # Initialize state
    state = {
        "messages": [HumanMessage(content=query)],
        "category": "",
        "response": ""
    }
    
    # Execute the graph
    try:
        result = graph.invoke(state)
        
        # Return the final response
        return {
            "category": result["category"],
            "response": result["response"]
        }
    except Exception as e:
        # Handle any errors in the graph execution
        return {
            "category": "other",
            "response": f"Error processing query: {str(e)}"
        }

# Test the routing system
if __name__ == "__main__":
    test_inputs = [
        "How do I install your Python library?",
        "What's the price of your enterprise plan?",
        "My account is locked and I can't log in",
        "Tell me about your company",
        "I want to integrate your API with my application"
    ]
    
    for input_text in test_inputs:
        result = process_query(input_text)
        print(f"Input: '{input_text}'")
        print(f"Category: {result['category']}")
        print(f"Response: {result['response']}")
        print("-" * 70)