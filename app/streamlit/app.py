import os
import gc
import tempfile
import uuid
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import requests
import json
from datetime import datetime

# Add LLM and embedding imports
from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.readers.docling import DoclingReader
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.query_engine import RetrieverQueryEngine

import streamlit as st
from dotenv import load_dotenv
load_dotenv()


# API configuration
API_URL = "http://localhost:5000/api"

# Page configuration
st.set_page_config(
    page_title="Video Game Sales Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
    
session_id = st.session_state.id


# Function to process and format the response
def process_response(text):
    """Process the response text to improve formatting."""
    # Ensure code blocks are properly formatted
    if "```" in text:
        # Enhance code block styling
        text = text.replace("```python", '<pre><code class="language-python">')
        text = text.replace("```", '</code></pre>')

    # Enhance list formatting
    lines = text.split('\n')
    for i, line in enumerate(lines):
        # Improve bullet point appearance
        if line.strip().startswith('- '):
            lines[i] = '• ' + line.strip()[2:]
        # Enhance numbered lists
        elif line.strip() and line.strip()[0].isdigit() and '. ' in line:
            num, content = line.split('. ', 1)
            lines[i] = f"<strong>{num}.</strong> {content}"

    # Join the processed lines back into text
    return '\n'.join(lines)


# Add custom CSS for chat styling
def add_custom_css():
    st.markdown("""
    <style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #f0f2f6;
    }
    .chat-message.assistant {
        background-color: #e6f3ff;
        border-left: 5px solid #0078ff;
    }
    .blinking-cursor {
        display: inline-block;
        animation: blink 1s step-end infinite;
    }
    @keyframes blink {
        50% { opacity: 0; }
    }
    </style>
    """, unsafe_allow_html=True)

add_custom_css()

# Load LLM model
@st.cache_resource
def load_llm():
    return Groq(
        model="llama3-8b-8192",
        api_key=os.getenv("groq_api_key"),
        max_tokens=4000,
        context_window=8192,
        temperature=0.1
    )

# Load embedding model
@st.cache_resource
def load_embedding_model():
    # Improved handling of Hugging Face API key
    hf_api_key = os.getenv("HF_API_KEY")
    if hf_api_key:
        os.environ["HF_API_KEY"] = hf_api_key
        os.environ["HUGGINGFACE_API_KEY"] = hf_api_key  # Some HF libraries use this variable name
    
    try:
        if hf_api_key:
            # Try to load the preferred model first
            return HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
            trust_remote_code=False,  # For security
            embed_batch_size=10
            )
            
        else:
            raise ValueError("No HF API key provided")
    except Exception as e:
        st.warning(f"Failed to load Hugging Face embedding models: {str(e)}")
        
        # # Use a completely local embedding model from llama_index
        # from llama_index.embeddings.ollama import OllamaEmbedding
        # try:
        #     return OllamaEmbedding(model_name="llama3", base_url="http://localhost:11434")
        # except Exception as ollama_err:
        #     st.warning(f"Failed to load Ollama embedding: {str(ollama_err)}")
            
        #     # Final fallback to the most basic embedding
        #     from llama_index.embeddings.fastembed import FastEmbedEmbedding
        #     return FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Reset chat history
def reset_chat():
    if "chat_history" in st.session_state:
        st.session_state.chat_history = []
    if "messages" in st.session_state:
        st.session_state.messages = []
    if "context" in st.session_state:
        st.session_state.context = None
    gc.collect()

# Function to display Excel files
def display_excel(file):
    st.markdown("### Excel Preview")
    # Read the Excel file
    df = pd.read_excel(file)
    # Display the dataframe
    st.dataframe(df)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .card {
        border-radius: 5px;
        background-color: #f9f9f9;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 1rem;
        color: #757575;
    }
    .stChat message {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# API Helper Functions
def fetch_api_data(endpoint):
    """Fetch data from API endpoint"""
    try:
        response = requests.get(f"{API_URL}/{endpoint}")
        return response.json()
    except Exception as e:
        st.error(f"Error fetching data from API: {e}")
        return None

# Sidebar
with st.sidebar:
    st.title("Video Games Sales Analysis")
    
    # Navigation
    st.header("Navigation")
    page = st.radio("Go to", ["Overview", "Platform Analysis", "Genre Analysis", 
                             "Publisher Analysis", "Geographic Analysis", 
                             "ESRB Rating Analysis", "Sunburst Visualizations", 
                             "Chat with Data", ])
    
    # Document Upload Section (only for RAG Analysis page)
    # if page == "RAG Analysis":
    #     st.header("Add your documents")
    #     uploaded_file = st.file_uploader("Choose your `.xlsx` file", type=["xlsx", "xls", "csv"])

    #     if uploaded_file:
    #         try:
    #             with tempfile.TemporaryDirectory() as temp_dir:
    #                 file_path = os.path.join(temp_dir, uploaded_file.name)
                    
    #                 with open(file_path, "wb") as f:
    #                     f.write(uploaded_file.getvalue())
                    
    #                 file_key = f"{session_id}-{uploaded_file.name}"
    #                 st.write("Indexing your document...")

    #                 if file_key not in st.session_state.get('file_cache', {}):
    #                     if os.path.exists(temp_dir):
    #                         if uploaded_file.name.endswith(('.xlsx', '.xls')):
    #                             reader = DoclingReader()
    #                             loader = SimpleDirectoryReader(
    #                                 input_dir=temp_dir,
    #                                 file_extractor={".xlsx": reader, ".xls": reader},
    #                             )
    #                         else:
    #                             loader = SimpleDirectoryReader(
    #                                 input_dir=temp_dir
    #                             )
    #                     else:    
    #                         st.error('Could not find the file you uploaded, please check again...')
    #                         st.stop()
                        
    #                     docs = loader.load_data()

    #                     # setup llm & embedding model
    #                     llm = load_llm()
    #                     embed_model = load_embedding_model()
                        
    #                     # Creating an index over loaded data
    #                     Settings.embed_model = embed_model
    #                     node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    #                     index = VectorStoreIndex.from_documents(documents=docs, transformations=[node_parser], show_progress=True)

    #                     # Create the query engine, where we use a cohere reranker on the fetched nodes
    #                     retriever = index.as_retriever(similarity_top_k=3)
    #                     qa_prompt_tmpl = PromptTemplate(
    #                         "Context information is below.\n"
    #                         "---------------------\n"
    #                         "{context_str}\n"
    #                         "---------------------\n"
    #                         "Given the context information above I want you to think step by step to answer the query in a highly precise and crisp manner focused on the final answer, incase case you don't know the answer say 'I don't know!'.\n"
    #                         "Query: {query_str}\n"
    #                         "Answer: "
    #                     )

    #                     response_synthesizer = CompactAndRefine(
    #                         text_qa_template=qa_prompt_tmpl,
    #                         streaming=True
    #                     )

    #                     query_engine = RetrieverQueryEngine(
    #                         retriever=retriever,
    #                         response_synthesizer=response_synthesizer
    #                     )

    #                     st.session_state.file_cache[file_key] = query_engine
    #                 else:
    #                     query_engine = st.session_state.file_cache[file_key]

    #                 # Inform the user that the file is processed and Display the PDF uploaded
    #                 st.success("Ready to Chat!")
    #                 if uploaded_file.name.endswith(('.xlsx', '.xls')):
    #                     display_excel(uploaded_file)
    #                 else:
    #                     st.write(f"File loaded: {uploaded_file.name}")
                    
    #                 if "messages" not in st.session_state:
    #                     reset_chat()

    #                 for message in st.session_state.messages:
    #                     with st.chat_message(message["role"]):
    #                         st.markdown(message["content"])

    #                 if prompt := st.chat_input("Ask a question about your data..."):
    #                     st.session_state.messages.append({"role": "user", "content": prompt})
    #                     with st.chat_message("user"):
    #                         st.markdown(prompt)

    #                     with st.chat_message("assistant"):
    #                         message_placeholder = st.empty()
    #                         full_response = ""
    #                         with st.spinner("Thinking..."):
    #                             streaming_response = query_engine.query(prompt)
    #                             for chunk in streaming_response.response_gen:
    #                                 full_response += chunk
    #                                 formatted_response = full_response + '<span class="blinking-cursor">▌</span>'
    #                                 message_placeholder.markdown(formatted_response, unsafe_allow_html=True)
    #                         formatted_final = process_response(full_response)
    #                         message_placeholder.markdown(formatted_final, unsafe_allow_html=True)
    #                     st.session_state.messages.append({"role": "assistant", "content": full_response})
    #         except Exception as e:
    #             st.error(f"An error occurred: {e}")
    #             st.stop()

# Main app title
st.markdown("<h1 class='main-header'>Video Game Sales Analysis Dashboard</h1>", unsafe_allow_html=True)

# Load basic stats
stats = fetch_api_data("stats")

# Function to create metrics row
def create_metrics_row(stats):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{stats['total_records']:,}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Total Games</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{stats['year_range'][0]} - {stats['year_range'][1]}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Year Range</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{stats['platforms']}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Platforms</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col4:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{stats['genres']}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Genres</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Overview Page
if page == "Overview":
    # Display metrics
    create_metrics_row(stats)
    
    # About the dataset
    st.markdown("<h2 class='sub-header'>About the Dataset</h2>", unsafe_allow_html=True)
    st.write("""
    This dashboard provides insights into video game sales data across different platforms, genres, and regions.
    The dataset contains information about video games released from the early days of gaming up to recent years.
    
    Use the sidebar navigation to explore different aspects of the data.
    """)
    
    # Quick insights sections
    st.markdown("<h2 class='sub-header'>Quick Insights</h2>", unsafe_allow_html=True)
    
    # Get genre data for pie chart
    genre_data = fetch_api_data("genres")
    if genre_data:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h3>Top Genres by Sales</h3>", unsafe_allow_html=True)
            # Create a DataFrame for the pie chart
            genre_df = pd.DataFrame(genre_data["genre_data"])
            fig = px.pie(genre_df, values='Total_Sales', names='Genre', 
                        title='Distribution of Sales by Genre',
                        hover_data=['Total_Sales'],
                        labels={'Total_Sales': 'Sales (millions)'},
                        hole=0.3)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        # Get platform data for bar chart
        platform_data = fetch_api_data("platforms")
        if platform_data:
            with col2:
                st.markdown("<h3>Top Platforms by Sales</h3>", unsafe_allow_html=True)
                # Create a DataFrame for the bar chart
                platform_df = pd.DataFrame(platform_data["platform_sum"])
                fig = px.bar(platform_df, x='Platform', y='Total_Sales',
                            title='Total Sales by Platform (Millions)',
                            color='Total_Sales',
                            color_continuous_scale='blues')
                fig.update_layout(xaxis_title="Platform", yaxis_title="Sales (millions)")
                st.plotly_chart(fig, use_container_width=True)
    
    # Display a sample of recent games trend
    st.markdown("<h3>Sales Trend Over Time</h3>", unsafe_allow_html=True)
    
    geo_data = fetch_api_data("geographic")
    if geo_data:
        geo_df = pd.DataFrame(geo_data["geo_data"])
        
        # Convert Year_of_Release to int if it's a float
        geo_df['Year_of_Release'] = geo_df['Year_of_Release'].astype(int)
        
        # Filter out years before 1980
        geo_df = geo_df[geo_df['Year_of_Release'] >= 1980]
        
        # Create a line chart for total sales over time
        fig = px.line(geo_df, x='Year_of_Release', y='Total_Sales',
                    title='Total Video Game Sales by Year (Millions)',
                    markers=True)
        fig.update_layout(xaxis_title="Year", yaxis_title="Sales (millions)")
        st.plotly_chart(fig, use_container_width=True)

# Platform Analysis Page
elif page == "Platform Analysis":
    st.markdown("<h2 class='sub-header'>Platform Analysis</h2>", unsafe_allow_html=True)
    st.write("Explore video game sales across different gaming platforms.")
    
    # Get platform data
    platform_data = fetch_api_data("platforms")
    platform_counts = fetch_api_data("platform_counts")
    
    if platform_data and platform_counts:
        # Platform sales over time
        st.markdown("<h3>Platform Sales Over Time</h3>", unsafe_allow_html=True)
        
        # Create a DataFrame for the animation
        platform_sales_df = pd.DataFrame(platform_data["platform_sales"])
        platform_sales_df['Year_of_Release'] = platform_sales_df['Year_of_Release'].astype(int)
        
        # Filter years
        min_year = st.slider("Select Year Range", 1980, 2020, (1980, 2020))
        filtered_df = platform_sales_df[(platform_sales_df['Year_of_Release'] >= min_year[0]) & 
                                        (platform_sales_df['Year_of_Release'] <= min_year[1])]
        
        fig = px.bar(filtered_df,
                    x='Platform',
                    y='Total_Sales',
                    animation_frame='Year_of_Release',
                    title='Platform Sales by Year (Millions)',
                    range_y=[0, 150])
        fig.update_xaxes(type='category')
        fig.update_xaxes(categoryorder='category ascending')
        st.plotly_chart(fig, use_container_width=True)
        
        # Total sales by platform
        st.markdown("<h3>Total Sales by Platform</h3>", unsafe_allow_html=True)
        
        platform_sum_df = pd.DataFrame(platform_data["platform_sum"])
        fig = px.bar(platform_sum_df,
                    x='Platform',
                    y='Total_Sales',
                    title='Total Sales by Platform (Millions)',
                    color='Total_Sales',
                    color_continuous_scale='blues')
        fig.update_layout(xaxis_title="Platform", yaxis_title="Sales (millions)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Platform releases over recent years
        st.markdown("<h3>Game Releases by Platform (Recent Years)</h3>", unsafe_allow_html=True)
        
        platform_counts_df = pd.DataFrame(platform_counts)
        fig = px.bar(platform_counts_df,
                    x='Platform',
                    y='Count',
                    color='Year_of_Release',
                    title='Number of Games Released by Platform',
                    barmode='group')
        fig.update_layout(xaxis_title="Platform", yaxis_title="Number of Games Released")
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent platform performance
        st.markdown("<h3>Platform Performance (Recent Years)</h3>", unsafe_allow_html=True)
        
        platform_recent_df = pd.DataFrame(platform_data["platform_recent"])
        fig = px.bar(platform_recent_df,
                    x='Platform',
                    y='Total_Sales',
                    color='Year_of_Release',
                    title='Platform Sales in Recent Years (Millions)',
                    barmode='group')
        fig.update_layout(xaxis_title="Platform", yaxis_title="Sales (millions)")
        st.plotly_chart(fig, use_container_width=True)

# Genre Analysis Page
elif page == "Genre Analysis":
    st.markdown("<h2 class='sub-header'>Genre Analysis</h2>", unsafe_allow_html=True)
    st.write("Explore video game sales across different genres.")
    
    # Get genre data
    genre_data = fetch_api_data("genres")
    
    if genre_data:
        # Genre sales distribution
        st.markdown("<h3>Genre Sales Distribution</h3>", unsafe_allow_html=True)
        
        genre_df = pd.DataFrame(genre_data["genre_data"])
        
        # Pie chart
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(genre_df,
                        values='Total_Sales',
                        names='Genre',
                        title='Distribution of Sales by Genre',
                        hole=0.3)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        # Bar chart
        with col2:
            fig = px.bar(genre_df,
                        x='Genre',
                        y='Total_Sales',
                        title='Total Sales by Genre (Millions)',
                        color='Total_Sales',
                        color_continuous_scale='blues')
            fig.update_layout(xaxis_title="Genre", yaxis_title="Sales (millions)")
            st.plotly_chart(fig, use_container_width=True)
        
        # Genre sales by region
        st.markdown("<h3>Genre Sales by Region</h3>", unsafe_allow_html=True)
        
        genre_region_df = pd.DataFrame(genre_data["genre_by_region"])
        
        # Create a new DataFrame for the heatmap
        heatmap_data = genre_region_df.copy()
        heatmap_data = heatmap_data.drop('Total_Sales', axis=1)
        
        # Pivot for the heatmap
        pivot_df = pd.melt(heatmap_data, 
                        id_vars=['Genre'],
                        value_vars=['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'],
                        var_name='Region',
                        value_name='Sales')
        
        # Clean up region names
        pivot_df['Region'] = pivot_df['Region'].apply(lambda x: x.replace('_Sales', ''))
        
        fig = px.density_heatmap(pivot_df,
                                x='Region',
                                y='Genre',
                                z='Sales',
                                title='Sales by Genre and Region (Millions)',
                                color_continuous_scale='blues')
        fig.update_layout(xaxis_title="Region", yaxis_title="Genre")
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent years genre analysis
        st.markdown("<h3>Genre Performance in Recent Years (2016-2019)</h3>", unsafe_allow_html=True)
        
        genre_recent_df = pd.DataFrame(genre_data["genre_recent"])
        
        # Create a new DataFrame for the comparison
        genre_comparison = pd.DataFrame({
            'Genre': genre_df['Genre'],
            'All Time': genre_df['Total_Sales']
        })
        
        # Get genres in recent data
        recent_genres = genre_recent_df['Genre'].tolist()
        
        # Add recent sales with matching
        genre_comparison['Recent Years'] = genre_comparison['Genre'].apply(
            lambda x: genre_recent_df.loc[genre_recent_df['Genre'] == x, 'Total_Sales'].values[0] 
            if x in recent_genres else 0
        )
        
        # Calculate percentage change
        genre_comparison['Percentage'] = ((genre_comparison['Recent Years'] / genre_comparison['All Time']) * 100).round(2)
        
        # Sort by all-time sales
        genre_comparison = genre_comparison.sort_values('All Time', ascending=False)
        
        # Create a grouped bar chart for comparison
        comparison_df = pd.melt(genre_comparison, 
                            id_vars=['Genre', 'Percentage'],
                            value_vars=['All Time', 'Recent Years'],
                            var_name='Period',
                            value_name='Sales')
        
        fig = px.bar(comparison_df,
                    x='Genre',
                    y='Sales',
                    color='Period',
                    title='Genre Sales: All Time vs Recent Years (Millions)',
                    barmode='group',
                    hover_data=['Percentage'])
        fig.update_layout(xaxis_title="Genre", yaxis_title="Sales (millions)")
        st.plotly_chart(fig, use_container_width=True)

# Publisher Analysis Page
elif page == "Publisher Analysis":
    st.markdown("<h2 class='sub-header'>Publisher Analysis</h2>", unsafe_allow_html=True)
    st.write("Explore video game sales by publisher.")
    
    # Get publisher data
    publisher_data = fetch_api_data("publishers")
    
    if publisher_data:
        # Top publishers
        st.markdown("<h3>Top Game Publishers</h3>", unsafe_allow_html=True)
        
        publishers_df = pd.DataFrame(publisher_data["publishers"])
        
        # Select top N publishers by sales
        top_n = st.slider("Select number of top publishers to display", 5, 20, 10)
        top_publishers_df = publishers_df.sort_values('Sales_Sum', ascending=False).head(top_n)
        
        fig = px.bar(top_publishers_df,
                    x='Publisher',
                    y='Sales_Sum',
                    title=f'Top {top_n} Publishers by Total Sales (Millions)',
                    color='Sales_Sum',
                    color_continuous_scale='blues')
        fig.update_layout(xaxis_title="Publisher", yaxis_title="Sales (millions)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Publisher comparison: Sales vs Number of Games
        st.markdown("<h3>Publisher Comparison: Sales vs Number of Games</h3>", unsafe_allow_html=True)
        
        fig = px.scatter(publishers_df,
                        x='Sales_Count',
                        y='Sales_Sum',
                        size='Sales_Sum',
                        color='Sales_Sum',
                        hover_name='Publisher',
                        title='Publishers: Number of Games vs Total Sales',
                        log_x=True)
        fig.update_layout(xaxis_title="Number of Games (log scale)", yaxis_title="Total Sales (millions)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Top 5 publishers and their genres
        st.markdown("<h3>Top 5 Publishers and Their Genre Focus</h3>", unsafe_allow_html=True)
        
        # Get publisher-genre data
        pub_genre_df = pd.DataFrame(publisher_data["pub_genre"])
        
        # Only use data for top 5 publishers
        top_5_publishers = publisher_data["top_publishers"]
        
        # Create a pivot table
        pub_genre_pivot = pub_genre_df.pivot_table(
            index='Publisher',
            columns='Genre',
            values='Total_Sales',
            aggfunc='sum'
        ).reset_index()
        
        # Convert to long format for visualization
        pub_genre_long = pd.melt(
            pub_genre_pivot,
            id_vars=['Publisher'],
            var_name='Genre',
            value_name='Sales'
        )
        
        # Only show data for top 5 publishers
        pub_genre_long = pub_genre_long[pub_genre_long['Publisher'].isin(top_5_publishers)]
        
        fig = px.bar(pub_genre_long,
                    x='Publisher',
                    y='Sales',
                    color='Genre',
                    title='Genre Focus of Top 5 Publishers (Sales in Millions)',
                    barmode='stack')
        fig.update_layout(xaxis_title="Publisher", yaxis_title="Sales (millions)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap of publishers and genres
        st.markdown("<h3>Publisher-Genre Heatmap</h3>", unsafe_allow_html=True)
        
        # Create a pivot table for the heatmap
        heatmap_df = pub_genre_df.pivot_table(
            index='Publisher',
            columns='Genre',
            values='Total_Sales',
            aggfunc='sum'
        )
        
        # Only show data for top publishers and fill NaN with 0
        heatmap_df = heatmap_df.loc[top_5_publishers].fillna(0)
        
        fig = px.imshow(heatmap_df,
                        title='Sales by Publisher and Genre (Millions)',
                        color_continuous_scale='blues',
                        text_auto='.1f')
        fig.update_layout(xaxis_title="Genre", yaxis_title="Publisher")
        st.plotly_chart(fig, use_container_width=True)

# Geographic Analysis Page
elif page == "Geographic Analysis":
    st.markdown("<h2 class='sub-header'>Geographic Analysis</h2>", unsafe_allow_html=True)
    st.write("Explore video game sales across different geographic regions.")
    
    # Get geographic data
    geo_data = fetch_api_data("geographic")
    
    if geo_data:
        # Total sales by region over time
        st.markdown("<h3>Sales by Region Over Time</h3>", unsafe_allow_html=True)
        
        geo_df = pd.DataFrame(geo_data["geo_data"])
        
        # Convert Year_of_Release to int
        geo_df['Year_of_Release'] = geo_df['Year_of_Release'].astype(int)
        
        # Filter out extreme years
        geo_df = geo_df[(geo_df['Year_of_Release'] >= 1980) & (geo_df['Year_of_Release'] <= 2020)]
        
        # Create line chart
        fig = go.Figure()
        
        for region in geo_data["regions"]:
            fig.add_trace(go.Scatter(
                x=geo_df['Year_of_Release'],
                y=geo_df[f'{region}_Sales'],
                mode='lines',
                name=region
            ))
        
        fig.update_layout(
            title='Total Sales per Year by Region (Millions)',
            xaxis_title='Year',
            yaxis_title='Sales (millions)',
            xaxis=dict(type='category')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Sales distribution by region
        st.markdown("<h3>Sales Distribution by Region</h3>", unsafe_allow_html=True)
        
        # Calculate total sales by region
        region_totals = {
            'Region': [],
            'Sales': []
        }
        
        for region in geo_data["regions"]:
            region_totals['Region'].append(region)
            region_totals['Sales'].append(geo_df[f'{region}_Sales'].sum())
        
        region_totals_df = pd.DataFrame(region_totals)
        
        # Create pie chart
        fig = px.pie(region_totals_df,
                    values='Sales',
                    names='Region',
                    title='Distribution of Global Sales by Region',
                    hole=0.3)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Regional sales trends
        st.markdown("<h3>Regional Sales Trends</h3>", unsafe_allow_html=True)
        
        # Create a DataFrame for the animation
        region_trends = []
        
        for region in geo_data["regions"]:
            for _, row in geo_df.iterrows():
                region_trends.append({
                    'Year': row['Year_of_Release'],
                    'Region': region,
                    'Sales': row[f'{region}_Sales']
                })
        
        region_trends_df = pd.DataFrame(region_trends)
        
        # Create animated bar chart
        fig = px.bar(region_trends_df,
                    x='Region',
                    y='Sales',
                    animation_frame='Year',
                    title='Regional Sales by Year (Millions)',
                    color='Region',
                    range_y=[0, 350])
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Regional market share over time
        st.markdown("<h3>Regional Market Share Over Time</h3>", unsafe_allow_html=True)
        
        # Calculate market share percentages
        market_share = []
        
        for _, row in geo_df.iterrows():
            year = row['Year_of_Release']
            total = row['Total_Sales']
            
            if total > 0:  # Avoid division by zero
                for region in geo_data["regions"]:
                    market_share.append({
                        'Year': year,
                        'Region': region,
                        'Percentage': (row[f'{region}_Sales'] / total) * 100
                    })
        
        market_share_df = pd.DataFrame(market_share)
        
        # Create area chart
        fig = px.area(market_share_df,
                    x='Year',
                    y='Percentage',
                    color='Region',
                    title='Regional Market Share Over Time (%)',
                    groupnorm='percent')
        fig.update_layout(xaxis_title='Year', yaxis_title='Market Share (%)')
        
        st.plotly_chart(fig, use_container_width=True)

# ESRB Rating Analysis Page
elif page == "ESRB Rating Analysis":
    st.markdown("<h2 class='sub-header'>ESRB Rating Analysis</h2>", unsafe_allow_html=True)
    st.write("Explore video game sales by ESRB rating.")
    
    # Get rating data
    rating_data = fetch_api_data("ratings")
    
    if rating_data and "message" not in rating_data:
        # Sales by rating
        st.markdown("<h3>Sales by ESRB Rating</h3>", unsafe_allow_html=True)
        
        ratings_df = pd.DataFrame(rating_data["ratings"])
        
        fig = px.bar(ratings_df,
                    x='Rating',
                    y='Total_Sales',
                    title='Total Sales by ESRB Rating (Millions)',
                    color='Total_Sales',
                    color_continuous_scale='blues')
        fig.update_layout(xaxis_title="ESRB Rating", yaxis_title="Sales (millions)")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Rating distribution by region
        st.markdown("<h3>ESRB Rating Sales by Region</h3>", unsafe_allow_html=True)
        
        # Create a DataFrame for the heatmap
        rating_region_df = pd.DataFrame(rating_data["rating_by_region"])
        
        # Create a new DataFrame for the heatmap
        heatmap_data = rating_region_df.copy()
        heatmap_data = heatmap_data.drop('Total_Sales', axis=1)
        
        # Pivot for the heatmap
        pivot_df = pd.melt(heatmap_data, 
                        id_vars=['Rating'],
                        value_vars=['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'],
                        var_name='Region',
                        value_name='Sales')
        
        # Clean up region names
        pivot_df['Region'] = pivot_df['Region'].apply(lambda x: x.replace('_Sales', ''))
        
        fig = px.density_heatmap(pivot_df,
                                x='Region',
                                y='Rating',
                                z='Sales',
                                title='Sales by ESRB Rating and Region (Millions)',
                                color_continuous_scale='blues')
        fig.update_layout(xaxis_title="Region", yaxis_title="ESRB Rating")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Rating distribution by genre
        st.markdown("<h3>ESRB Rating Distribution by Genre</h3>", unsafe_allow_html=True)
        
        # Create a DataFrame for the heatmap
        rating_genre_df = pd.DataFrame(rating_data["rating_by_genre"])
        
        # Create a pivot table
        rating_genre_pivot = rating_genre_df.pivot_table(
            index='Rating',
            columns='Genre',
            values='Total_Sales',
            aggfunc='sum'
        ).fillna(0)
        
        # Create heatmap
        fig = px.imshow(rating_genre_pivot,
                        title='Sales by ESRB Rating and Genre (Millions)',
                        color_continuous_scale='blues',
                        text_auto='.1f')
        fig.update_layout(xaxis_title="Genre", yaxis_title="ESRB Rating")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Rating pie chart
        st.markdown("<h3>ESRB Rating Sales Distribution</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(ratings_df,
                        values='Total_Sales',
                        names='Rating',
                        title='Distribution of Sales by ESRB Rating',
                        hole=0.3)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Calculate total games by rating
            games_by_rating = rating_region_df.groupby('Rating').size().reset_index(name='count')
            
            fig = px.pie(games_by_rating,
                        values='count',
                        names='Rating',
                        title='Distribution of Games by ESRB Rating',
                        hole=0.3)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ESRB Rating data is not available in this dataset.")

# Sunburst Visualizations Page
elif page == "Sunburst Visualizations":
    st.markdown("<h2 class='sub-header'>Multi-dimensional Analysis</h2>", unsafe_allow_html=True)
    st.write("Explore the relationships between platforms, genres, and publishers using interactive sunburst visualizations.")
    
    # Get sunburst data
    sunburst_data = fetch_api_data("sunburst")
    
    if sunburst_data:
        # Platform and Genre Sunburst
        st.markdown("<h3>Platform and Genre Relationship</h3>", unsafe_allow_html=True)
        
        plat_genre_df = pd.DataFrame(sunburst_data["platform_genre"])
        
        fig = px.sunburst(plat_genre_df,
                        path=['Genre', 'Platform'],
                        values='Total_Sales',
                        title='Sales Distribution by Genre and Platform')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Genre and Publisher Sunburst
        st.markdown("<h3>Genre and Publisher Relationship</h3>", unsafe_allow_html=True)
        
        genre_pub_df = pd.DataFrame(sunburst_data["genre_publisher"])
        
        fig = px.sunburst(genre_pub_df,
                        path=['Genre', 'Publisher'],
                        values='Total_Sales',
                        title='Sales Distribution by Genre and Publisher')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Platform and Publisher Sunburst
        st.markdown("<h3>Platform and Publisher Relationship</h3>", unsafe_allow_html=True)
        
        plat_pub_df = pd.DataFrame(sunburst_data["platform_publisher"])
        
        fig = px.sunburst(plat_pub_df,
                        path=['Platform', 'Publisher'],
                        values='Total_Sales',
                        title='Sales Distribution by Platform and Publisher')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # All Together Sunburst
        st.markdown("<h3>Genre, Platform, and Publisher Relationship</h3>", unsafe_allow_html=True)
        
        all_together_df = pd.DataFrame(sunburst_data["all_together"])
        
        fig = px.sunburst(all_together_df,
                        path=['Genre', 'Platform', 'Publisher'],
                        values='Total_Sales',
                        title='Sales Distribution by Genre, Platform, and Publisher')
        
        st.plotly_chart(fig, use_container_width=True)

# Chat with Data Page
elif page == "Chat with Data":
    st.markdown("<h2 class='sub-header'>Chat with Your Data</h2>", unsafe_allow_html=True)
    st.write("Ask questions about video game sales data and get AI-powered insights.")
    
    # Create a container for the chat interface
    chat_container = st.container()
    
    # Initialize chat history in session state if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Function to generate AI response based on user query
    def generate_response(query):
        """Generate a response to the user's query by analyzing the video game sales data"""
        try:
            # Try to understand what kind of data the user is looking for
            if "platform" in query.lower() or "console" in query.lower():
                platform_data = fetch_api_data("platforms")
                if platform_data:
                    data = pd.DataFrame(platform_data["platform_sum"])
                    result = f"Based on my analysis of the platform data:\n\n"
                    result += f"The top selling platform is {data.iloc[0]['Platform']} with {data.iloc[0]['Total_Sales']:.1f} million in sales.\n"
                    result += f"The dataset includes {len(data)} major platforms with total sales exceeding 3% of the market.\n\n"
                    
                    # Add some specific insights
                    return result + "Would you like to know more details about a specific platform's performance?"
            
            elif "genre" in query.lower() or "type" in query.lower():
                genre_data = fetch_api_data("genres")
                if genre_data:
                    data = pd.DataFrame(genre_data["genre_data"])
                    result = f"Looking at the genre data:\n\n"
                    result += f"The most popular genre is {data.iloc[0]['Genre']} with {data.iloc[0]['Total_Sales']:.1f} million in sales.\n"
                    result += f"The top 3 genres are: {', '.join(data['Genre'].head(3).tolist())}.\n\n"
                    
                    return result + "Would you like more specific genre analysis?"
            
            elif "publisher" in query.lower() or "company" in query.lower() or "developer" in query.lower():
                publisher_data = fetch_api_data("publishers")
                if publisher_data:
                    data = pd.DataFrame(publisher_data["publishers"])
                    data = data.sort_values('Sales_Sum', ascending=False)
                    result = f"Regarding game publishers:\n\n"
                    result += f"The top publisher is {data.iloc[0]['Publisher']} with {data.iloc[0]['Sales_Sum']:.1f} million in sales and {int(data.iloc[0]['Sales_Count'])} games.\n"
                    result += f"The top 3 publishers are: {', '.join(data['Publisher'].head(3).tolist())}.\n\n"
                    
                    return result + "Is there a specific publisher you'd like to know more about?"
            
            elif "region" in query.lower() or "country" in query.lower() or "geographic" in query.lower():
                geo_data = fetch_api_data("geographic")
                if geo_data:
                    regions = geo_data["regions"]
                    geo_df = pd.DataFrame(geo_data["geo_data"])
                    
                    # Calculate total sales per region
                    region_sales = {}
                    for region in regions:
                        region_sales[region] = geo_df[f"{region}_Sales"].sum()
                    
                    # Get the region with the highest sales
                    top_region = max(region_sales, key=region_sales.get)
                    
                    result = f"Looking at regional sales data:\n\n"
                    result += f"The region with the highest total sales is {top_region} with {region_sales[top_region]:.1f} million in sales.\n"
                    result += f"Region breakdown: " + ", ".join([f"{r}: {region_sales[r]:.1f}M" for r in regions]) + "\n\n"
                    
                    return result + "Would you like to see how regional sales have changed over time?"
            
            elif "rating" in query.lower() or "esrb" in query.lower() or "age" in query.lower():
                rating_data = fetch_api_data("ratings")
                if rating_data and "message" not in rating_data:
                    data = pd.DataFrame(rating_data["ratings"])
                    data = data.sort_values('Total_Sales', ascending=False)
                    result = f"Regarding ESRB ratings:\n\n"
                    result += f"The rating with the highest sales is {data.iloc[0]['Rating']} with {data.iloc[0]['Total_Sales']:.1f} million in sales.\n"
                    result += f"The top 3 ratings by sales are: {', '.join(data['Rating'].head(3).tolist())}.\n\n"
                    
                    return result + "Would you like to know more about how ratings affect sales in specific regions?"
            
            elif "compare" in query.lower() or "versus" in query.lower() or " vs " in query.lower():
                return "I can compare different platforms, genres, publishers, or regions. Please specify what you'd like to compare, for example: 'Compare PS4 vs Xbox One' or 'Compare Action vs Sports genres'."
            
            elif "best" in query.lower() or "top" in query.lower() or "highest" in query.lower():
                if "year" in query.lower():
                    geo_data = fetch_api_data("geographic")
                    if geo_data:
                        geo_df = pd.DataFrame(geo_data["geo_data"])
                        geo_df['Year_of_Release'] = geo_df['Year_of_Release'].astype(int)
                        
                        # Find the year with the highest sales
                        best_year = geo_df.loc[geo_df['Total_Sales'].idxmax()]
                        
                        result = f"Based on total sales:\n\n"
                        result += f"The best year for video game sales was {int(best_year['Year_of_Release'])} with {best_year['Total_Sales']:.1f} million in sales.\n\n"
                        
                        return result + "Is there a specific time period you'd like to analyze?"
                else:
                    return "I can tell you about the best-selling games, platforms, genres, or publishers. Please specify what you're interested in, for example: 'What are the top 5 platforms?' or 'What's the best-selling genre?'"
            
            # If the query is complex or doesn't match simple patterns, provide a general response
            stats = fetch_api_data("stats")
            if stats:
                return f"I can help you analyze video game sales data covering {stats['total_records']} games from {stats['year_range'][0]} to {stats['year_range'][1]}. The dataset includes {stats['platforms']} platforms, {stats['genres']} genres, and {stats['publishers']} publishers. Please ask a more specific question about platforms, genres, publishers, regions, or ratings."
            
            return "I'm not sure how to answer that. Try asking about platforms, genres, publishers, regions, or ratings in the video game sales data."
        
        except Exception as e:
            return f"I encountered an error analyzing that data: {str(e)}. Please try a different question."
    
    # Display chat history
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    # Chat input
    user_query = st.chat_input("Ask a question about video game sales data...")
    
    if user_query:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Display user message
        with chat_container:
            with st.chat_message("user"):
                st.write(user_query)
        
        # Generate and display response
        with st.spinner("Analyzing data..."):
            response = generate_response(user_query)
            
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Display assistant response
        with chat_container:
            with st.chat_message("assistant"):
                st.write(response)
    
    # Add some example queries to help users
    with st.expander("Example questions you can ask", expanded=True):
        st.markdown("""
        - What are the top selling platforms?
        - Which genre has the highest sales?
        - Who are the leading publishers in the industry?
        - How do sales compare across different regions?
        - What games have the highest ESRB ratings?
        - What was the best year for video game sales?
        - Compare PlayStation vs Xbox sales
        - Which genres are most popular in Japan vs North America?
        """)

# Footer with timestamp
st.markdown(f"""
<div style="text-align: center; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #e0e0e0;">
    <p>Video Game Sales Analysis Dashboard</p>
</div>
""", unsafe_allow_html=True)