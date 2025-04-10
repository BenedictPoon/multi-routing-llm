"""
Simple rule-based classifier for demonstration purposes
"""

def classify_input(user_input):
    """
    Classify the user input into one of the predefined categories based on keywords.
    
    Args:
        user_input (str): The user's query to classify
        
    Returns:
        str: The classified category
    """
    user_input = user_input.lower()
    
    # Define keyword groups for each category
    technical_keywords = [
        "install", "setup", "configure", "api", "code", "programming", 
        "library", "function", "error", "bug", "documentation", "sdk",
        "implementation", "integrate", "python", "javascript", "java"
    ]
    
    product_keywords = [
        "price", "cost", "subscription", "plan", "pricing", "trial", 
        "free", "premium", "enterprise", "features", "compare", "offer",
        "discount", "package", "upgrade", "downgrade", "license"
    ]
    
    support_keywords = [
        "help", "account", "locked", "password", "reset", "login", 
        "access", "denied", "problem", "issue", "can't", "unable", 
        "trouble", "fix", "broken", "not working", "error"
    ]
    
    general_keywords = [
        "about", "company", "who", "what", "when", "where", "why", 
        "how", "information", "contact", "team", "history", "mission",
        "vision", "values", "founded", "headquarters", "location"
    ]
    
    # Check for keyword matches in each category
    technical_score = sum(1 for keyword in technical_keywords if keyword in user_input)
    product_score = sum(1 for keyword in product_keywords if keyword in user_input)
    support_score = sum(1 for keyword in support_keywords if keyword in user_input)
    general_score = sum(1 for keyword in general_keywords if keyword in user_input)
    
    # Determine the category with the highest score
    scores = {
        "technical_question": technical_score,
        "product_inquiry": product_score,
        "customer_support": support_score,
        "general_inquiry": general_score
    }
    
    # Find the category with the highest score
    max_category = max(scores, key=scores.get)
    
    # If no matches or a tie, return "other"
    if scores[max_category] == 0:
        return "other"
    
    return max_category

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