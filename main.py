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
    """Check if OpenAI API key is set in environment variables or config file"""
    api_key = os.environ.get('OPENAI_API_KEY')
    
    # If not in environment, try to load from config file
    if not api_key:
        try:
            config_path = os.path.join(os.path.dirname(__file__), 'config.txt')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    api_key = f.read().strip()
                os.environ["OPENAI_API_KEY"] = api_key
                print("API key loaded from config file.")
                return True
        except Exception as e:
            print(f"Error loading config: {e}")
    
    # If still no API key, ask for it
    if not api_key:
        print("OpenAI API key not found.")
        print("Please enter your OpenAI API key:")
        entered_key = input().strip()
        
        if entered_key:
            # Set the API key in environment variables
            os.environ["OPENAI_API_KEY"] = entered_key
            
            # Save to config file for future use
            try:
                config_path = os.path.join(os.path.dirname(__file__), 'config.txt')
                with open(config_path, 'w') as f:
                    f.write(entered_key)
                print("API key saved to config file.")
            except Exception as e:
                print(f"Error saving config: {e}")
                
            return True
        else:
            print("No API key provided.")
            return False
    
    print("Using OpenAI API key from environment.")
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