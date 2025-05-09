import os
import sys
import subprocess
import threading
import time
import webbrowser

def start_flask_api():
    """Start the Flask API server"""
    print("Starting Flask API server...")
    api_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "api", "app.py")
    
    # Use DEVNULL instead of PIPE to avoid buffer issues
    subprocess.Popen([sys.executable, api_path], 
                     stdout=subprocess.DEVNULL, 
                     stderr=subprocess.DEVNULL)
    
def start_streamlit_app():
    """Start the Streamlit app"""
    print("Starting Streamlit app...")
    streamlit_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "streamlit", "app.py")
    
    # Use DEVNULL instead of PIPE to avoid buffer issues
    subprocess.Popen([sys.executable, "-m", "streamlit", "run", streamlit_path],
                     stdout=subprocess.DEVNULL, 
                     stderr=subprocess.DEVNULL)

def main():
    """Start both the Flask API and Streamlit app"""
    print("Starting Video Game Sales Analysis Dashboard")
    
    # Start Flask API in a separate thread
    flask_thread = threading.Thread(target=start_flask_api)
    flask_thread.daemon = True
    flask_thread.start()
    
    # Give Flask API some time to start
    print("Waiting for Flask API to initialize...")
    time.sleep(5)
    
    # Start Streamlit app
    start_streamlit_app()
    
    # Open browser
    print("Opening browser...")
    webbrowser.open("http://localhost:8501")
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        sys.exit(0)

if __name__ == "__main__":
    main()