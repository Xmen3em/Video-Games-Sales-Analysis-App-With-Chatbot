# Video Games Sales Analysis App With Chatbot

An interactive dashboard application for analyzing video game sales data across different platforms, genres, publishers, and regions, with an integrated chatbot for data queries.

## Overview

This application provides comprehensive analysis and visualization of video game sales data, allowing users to explore trends and patterns across various dimensions of the gaming industry. The dashboard offers multiple analysis views and an interactive chatbot feature to query the data.

## Features


- **Overview Dashboard**: Get a quick summary of key metrics and trends in the video game industry.
- **Platform Analysis**: Explore video game sales across different gaming platforms.
- **Genre Analysis**: Analyze the popularity and sales performance of different game genres.
- **Publisher Analysis**: See which publishers dominate the market and their focus areas.
- **Geographic Analysis**: Visualize sales distributions across different regions (NA, EU, JP, Other).
- **ESRB Rating Analysis**: Understand how game ratings impact sales.
- **Multi-dimensional Analysis**: Interactive sunburst visualizations to explore relationships between platforms, genres, and publishers.
- **Chat with Data**: Ask questions about the data and get AI-powered responses

## Project Structure

```
Video-Games-Sales-Analysis-App-With-Chatbot/
│
├── app/                          # Main application code
│   ├── api/                      # Flask API backend
│   │   └── app.py                # Flask API application
│   ├── streamlit/                # Streamlit frontend
│   │   └── app.py                # Streamlit UI application
│   └── utils/                    # Utility modules
│       └── data_processor.py     # Data processing module
│
├── video_games_sales.csv         # Dataset
├── requirements.txt              # Dependencies
├── run_app.py                    # Main entry point to run the application
└── README.md                     # Project documentation
└── video games.ipynb
```

## Technology Stack

- **Backend**: Flask API
- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Data Analysis**: Object-Oriented Python
- **AI/ML**: LLaMA Index, Groq, HuggingFace Embeddings

## Installation

1. Clone the repository or download the source code.

    ```
    git clone https://github.com/Xmen3em/Video-Games-Sales-Analysis-App-With-Chatbot.git  
    cd Video-Games-Sales-Analysis-App-With-Chatbot  
    ```


2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

Run the entire application (both API and dashboard) with a single command:

```
python run_app.py
```

This will:
1. Start the Flask API backend on port 5000
2. Start the Streamlit frontend on port 8501
3. Open your browser to the dashboard automatically

Alternatively, you can run the components separately:

- To run just the Flask API:
  ```
  python app/api/app.py
  ```

- To run just the Streamlit dashboard (requires API to be running):
  ```
  streamlit run app/streamlit/app.py
  ```

## Insights and Analysis

The dashboard provides several key insights:

- Sales trends over time
- Platform popularity and lifecycle analysis
- Genre preference across different regions
- Publisher market share and specialization
- Regional market analysis
- Age rating impact on sales

## Dataset

The analysis is based on a video game sales dataset that includes information about games released from 1980 to 2020, covering:

- Game titles
- Platforms
- Year of release
- Genres
- Publishers
- Regional sales figures
- Age ratings

## System Architecture

1. The application follows a client-server architecture with clearly separated frontend and backend components:

2. Frontend Layer: Streamlit dashboard for visualization and user interaction

3. Backend Layer: Flask API that processes data requests and serves analysis results

4. Data Processing Layer: VideoGameAnalyzer class that handles all data analysis operations

5. Orchestration Layer: run_app.py script that coordinates all components


**Wiki pages you might want to explore:**

System Architecture [(Xmen3em/Video-Games-Sales-Analysis-App-With-Chatbot)](https://deepwiki.com/Xmen3em/Video-Games-Sales-Analysis-App-With-Chatbot/1.1-system-architecture)



