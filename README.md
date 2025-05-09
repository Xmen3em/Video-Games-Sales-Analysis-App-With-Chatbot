# Video Game Sales Analysis Dashboard

An interactive dashboard for analyzing video game sales data across different platforms, genres, publishers, and regions.

## Features

- **Platform Analysis**: Explore video game sales across different gaming platforms.
- **Genre Analysis**: Analyze the popularity and sales performance of different game genres.
- **Publisher Analysis**: See which publishers dominate the market and their focus areas.
- **Geographic Analysis**: Visualize sales distributions across different regions (NA, EU, JP, Other).
- **ESRB Rating Analysis**: Understand how game ratings impact sales.
- **Multi-dimensional Analysis**: Interactive sunburst visualizations to explore relationships between platforms, genres, and publishers.

## Project Structure

```
Video-Games-Sales/
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

## Installation

1. Clone the repository or download the source code.

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