import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Set up your OpenAI API key
# If you're using conda, you might want to add this to your .bashrc or .zshrc
# or you can uncomment and add your key here
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Initialize the language model
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Define categories for classification
CATEGORIES = ["technical_question", "product_inquiry", "customer_support", "general_inquiry", "other"]

# Create a classification prompt template
classification_prompt = ChatPromptTemplate.from_template("""
You are a classifier that categorizes user queries into predefined categories.
Based on the user input, determine the appropriate category from the following options:
{categories}

User input: {user_input}

Return ONLY the category name without any explanation or additional text.
""")

def classify_input(user_input):
    """
    Classify the user input into one of the predefined categories.
    
    Args:
        user_input (str): The user's query to classify
        
    Returns:
        str: The classified category
    """
    prompt = classification_prompt.format(
        categories="\n".join([f"- {category}" for category in CATEGORIES]),
        user_input=user_input
    )
    
    response = llm.invoke(prompt)
    category = response.content.strip().lower()
    
    # Ensure the returned category is valid
    if category not in CATEGORIES:
        # Default to 'other' if the model returns an invalid category
        category = "other"
    
    return category

# Test the classifier
if __name__ == "__main__":
    test_inputs = [
        "How do I install your Python library?",
        "What's the price of your enterprise plan?",
        "My account is locked and I can't log in",
        "Tell me about your company",
        "I want to integrate your API with my application"
    ]
    
    for input_text in test_inputs:
        category = classify_input(input_text)
        print(f"Input: '{input_text}'")
        print(f"Category: {category}")
        print("-" * 50)
