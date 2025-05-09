from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import sys

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_processor import VideoGameAnalyzer

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the analyzer with the data path
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                         'video_games_sales.csv')
analyzer = VideoGameAnalyzer(data_path)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "API is running"})

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get basic statistics about the dataset"""
    return jsonify(analyzer.get_basic_stats())

@app.route('/api/platforms', methods=['GET'])
def get_platforms():
    """Get platform sales data"""
    return jsonify(analyzer.get_platform_data())

@app.route('/api/platform_counts', methods=['GET'])
def get_platform_counts():
    """Get platform count data"""
    return jsonify(analyzer.get_platform_counts())

@app.route('/api/genres', methods=['GET'])
def get_genres():
    """Get genre data"""
    return jsonify(analyzer.get_genre_data())

@app.route('/api/geographic', methods=['GET'])
def get_geographic():
    """Get geographic sales data"""
    return jsonify(analyzer.get_geographic_data())

@app.route('/api/publishers', methods=['GET'])
def get_publishers():
    """Get publisher data"""
    return jsonify(analyzer.get_publisher_data())

@app.route('/api/ratings', methods=['GET'])
def get_ratings():
    """Get ESRB rating data"""
    return jsonify(analyzer.get_rating_data())

@app.route('/api/sunburst', methods=['GET'])
def get_sunburst():
    """Get sunburst visualization data"""
    return jsonify(analyzer.get_sunburst_data())

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)