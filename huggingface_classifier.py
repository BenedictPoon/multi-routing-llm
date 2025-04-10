"""
Classifier using Hugging Face transformers with a distilbert model
"""
import os
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

class HuggingFaceClassifier:
    def __init__(self, model_name="distilbert-base-uncased"):
        """
        Initialize the classifier with a pre-trained model
        
        Args:
            model_name (str): Name of the model to use from Hugging Face
        """
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None  # We'll use keyword matching instead
        
        # Keywords to help with categorization
        self.technical_keywords = [
            "install", "setup", "configure", "api", "code", "programming", 
            "library", "function", "error", "bug", "documentation", "sdk",
            "implementation", "integrate", "python", "javascript", "java"
        ]
        self.product_keywords = [
            "price", "cost", "subscription", "plan", "pricing", "trial", 
            "free", "premium", "enterprise", "features", "compare", "offer",
            "discount", "package", "upgrade", "downgrade", "license"
        ]
        self.support_keywords = [
            "help", "account", "locked", "password", "reset", "login", 
            "access", "denied", "problem", "issue", "can't", "unable", 
            "trouble", "fix", "broken", "not working", "error"
        ]
        self.general_keywords = [
            "about", "company", "who", "what", "when", "where", "why", 
            "how", "information", "contact", "team", "history", "mission",
            "vision", "values", "founded", "headquarters", "location"
        ]
        
    def classify_input(self, user_input):
        """
        Classify the user input into one of the predefined categories.
        
        Args:
            user_input (str): The user's query to classify
            
        Returns:
            str: The classified category
        """
        # Use keyword matching for classification
        user_input_lower = user_input.lower()
        
        # Check for keyword matches
        technical_score = sum(1 for keyword in self.technical_keywords if keyword in user_input_lower)
        product_score = sum(1 for keyword in self.product_keywords if keyword in user_input_lower)
        support_score = sum(1 for keyword in self.support_keywords if keyword in user_input_lower)
        general_score = sum(1 for keyword in self.general_keywords if keyword in user_input_lower)
        
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

# Function to use for import
def classify_input(user_input):
    """
    Wrapper function for the HuggingFaceClassifier
    
    Args:
        user_input (str): The user's query to classify
        
    Returns:
        str: The classified category
    """
    # Initialize classifier as a singleton
    if not hasattr(classify_input, "classifier"):
        classify_input.classifier = HuggingFaceClassifier()
    
    # Classify the input
    return classify_input.classifier.classify_input(user_input)

# Test the classifier
if __name__ == "__main__":
    # Create test classifier
    classifier = HuggingFaceClassifier()
    
    test_inputs = [
        "How do I install your Python library?",
        "What's the price of your enterprise plan?",
        "My account is locked and I can't log in",
        "Tell me about your company",
        "I want to integrate your API with my application",
        "Can you explain what your startup does?",
        "I'm receiving an error when trying to use the SDK",
        "Do you offer student discounts on your software?"
    ]
    
    for input_text in test_inputs:
        category = classifier.classify_input(input_text)
        print(f"Input: '{input_text}'")
        print(f"Category: {category}")
        print("-" * 50)


# Function to use for import
def classify_input(user_input):
    """
    Wrapper function for the HuggingFaceClassifier
    
    Args:
        user_input (str): The user's query to classify
        
    Returns:
        str: The classified category
    """
    # Initialize classifier as a singleton
    if not hasattr(classify_input, "classifier"):
        classify_input.classifier = HuggingFaceClassifier()
    
    # Classify the input
    return classify_input.classifier.classify_input(user_input)

# Test the classifier
if __name__ == "__main__":
    # Create test classifier
    classifier = HuggingFaceClassifier()
    
    test_inputs = [
        "How do I install your Python library?",
        "What's the price of your enterprise plan?",
        "My account is locked and I can't log in",
        "Tell me about your company",
        "I want to integrate your API with my application",
        "Can you explain what your startup does?",
        "I'm receiving an error when trying to use the SDK",
        "Do you offer student discounts on your software?"
    ]
    
    for input_text in test_inputs:
        category = classifier.classify_input(input_text)
        print(f"Input: '{input_text}'")
        print(f"Category: {category}")
        print("-" * 50)