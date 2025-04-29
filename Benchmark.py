import time
import classifier
import rule_based_classifier  

# Sample inputs to test
# Not functional at the moment, but we can use this file to create all the benchmarks we need.
test_inputs = [
    "How do I install your Python library?",
    "What's the price of your enterprise plan?",
    "My account is locked and I can't log in",
    "Tell me about your company",
    "I want to integrate your API with my application"
]

# Function for testing time
def time_benchmark_classifier(classifier_func, name):
    print(f"\n--- Benchmarking: {name} ---")
    start_time = time.time()
    
    for input_text in test_inputs:
        category = classifier_func(input_text)
        print(f"Input: {input_text}")
        print(f"Category: {category}")
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time for {name}: {total_time:.4f} seconds")
    print("-" * 60)

if __name__ == "__main__":
    time_benchmark_classifier(rule_based_classifier.classify_input, "Rule-Based Classifier")
    time_benchmark_classifier(classifier.classify_input, "LLM-Based Classifier")