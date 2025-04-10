import os
import subprocess
import sys

def check_environment():
    """Check if we're in the correct conda environment"""
    current_env = os.environ.get('CONDA_DEFAULT_ENV')
    
    if current_env != 'multillm':
        print(f"Warning: You are currently in '{current_env}' environment, not 'multillm'")
        print("Please activate the correct environment with: conda activate multillm")
        return False
    return True

def check_api_key():
    """Check if OpenAI API key is set in environment variables"""
    api_key = os.environ.get('OPENAI_API_KEY')
    
    if not api_key:
        print("OpenAI API key not found in environment variables.")
        print("Please enter your OpenAI API key:")
        entered_key = input().strip()
        
        if entered_key:
            # Set the API key in environment variables
            os.environ["OPENAI_API_KEY"] = entered_key
            print("API key set in environment variables.")
            
            # Also reload the router module to reinitialize the client
            import importlib
            import router
            importlib.reload(router)
            
            return True
        else:
            print("No API key provided.")
            return False
    else:
        print("Using OpenAI API key from environment variables.")
        return True

def run_streamlit():
    """Run the Streamlit application"""
    print("Starting MultiLLM Query Router...")
    try:
        subprocess.run(['streamlit', 'run', 'ui.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
    except FileNotFoundError:
        print("Streamlit not found. Please ensure it's installed with: pip install streamlit")

def main():
    """Main function to run the application"""
    # Check environment
    if not check_environment():
        user_input = input("Continue anyway? (y/n): ")
        if user_input.lower() != 'y':
            sys.exit(1)
    
    # Check API key
    if not check_api_key():
        user_input = input("Continue anyway? (y/n): ")
        if user_input.lower() != 'y':
            sys.exit(1)
    
    # Run the app
    run_streamlit()

if __name__ == "__main__":
    main()