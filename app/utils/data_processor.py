import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import json

class VideoGameAnalyzer:
    """Class for analyzing video game sales data"""
    
    def __init__(self, data_path):
        """Initialize with the path to the data file"""
        self.data_path = data_path
        self.df = None
        self.tdf = None
        self.load_data()
        self.preprocess_data()
        
    def load_data(self):
        """Load the data from CSV file"""
        self.df = pd.read_csv(self.data_path)
        return self.df.head().to_dict()
    
    def preprocess_data(self):
        """Preprocess the data for analysis"""
        # Create a copy of dataframe and handle missing values
        self.tdf = self.df.copy()
        
        # Filter out rows with missing year
        self.tdf = self.tdf[self.tdf['Year_of_Release'].notna()]
        
        # Sort by year
        self.tdf = self.tdf.sort_values('Year_of_Release', ascending=True)
        
        # Add Total_Sales column if not exists
        if 'Total_Shipped' in self.tdf.columns:
            self.tdf['Total_Sales'] = self.tdf['Total_Shipped'].fillna(0) + self.tdf['Global_Sales'].fillna(0)
        else:
            self.tdf['Total_Sales'] = self.tdf['Global_Sales']
            
        return {"message": "Data preprocessing complete", "rows": len(self.tdf)}
    
    def get_basic_stats(self):
        """Get basic statistics about the dataset"""
        stats = {
            "total_records": len(self.df),
            "cleaned_records": len(self.tdf),
            "year_range": [int(self.tdf['Year_of_Release'].min()), int(self.tdf['Year_of_Release'].max())],
            "platforms": self.tdf['Platform'].nunique(),
            "genres": self.tdf['Genre'].nunique(),
            "publishers": self.tdf['Publisher'].nunique(),
        }
        return stats
    
    def get_platform_data(self):
        """Get platform data for analysis"""
        platform_tdf = self.tdf.groupby(['Platform', 'Year_of_Release']).agg({'Total_Sales': 'sum'}).reset_index()
        platform_tdf = platform_tdf.sort_values('Year_of_Release', ascending=True)
        
        # Get top platforms by sales
        platform_sum_tdf = platform_tdf.groupby(['Platform']).agg({'Total_Sales': 'sum'}).reset_index()
        platform_sum_tdf = platform_sum_tdf[platform_sum_tdf['Total_Sales'] > platform_sum_tdf['Total_Sales'].sum() * 0.03]
        
        # Get recent years platform data (2016-2020)
        platform_top_tdf = platform_tdf[platform_tdf['Year_of_Release'].isin([2016, 2017, 2018, 2019, 2020])]
        platform_top_tdf = platform_top_tdf[platform_top_tdf['Total_Sales'] > platform_top_tdf['Total_Sales'].sum() * 0.005]
        platform_top_tdf['Year_of_Release'] = platform_top_tdf['Year_of_Release'].astype(str)
        
        return {
            "platform_sales": platform_tdf.to_dict(orient='records'),
            "platform_sum": platform_sum_tdf.to_dict(orient='records'),
            "platform_recent": platform_top_tdf.to_dict(orient='records')
        }
    
    def get_genre_data(self):
        """Get genre data for analysis"""
        # Define regions
        regions = ['NA', 'JP', 'EU', 'Other']
        region_sales_sufix = '_Sales'
        regions_agg = {}
        
        for region in regions:
            regions_agg[region + region_sales_sufix] = 'sum'
            
        regions_agg['Total_Sales'] = 'sum'
        
        # Genre analysis
        genre_tdf = self.tdf.groupby(['Genre']).agg(regions_agg)
        genre_tdf = genre_tdf.sort_values('Total_Sales', ascending=False)
        
        # Recent years genre analysis
        genre_last_tdf = self.tdf[self.tdf['Year_of_Release'].isin([2016, 2017, 2018, 2019])]
        genre_last_tdf = genre_last_tdf.groupby(['Genre']).agg(regions_agg)
        genre_last_tdf = genre_last_tdf.sort_values('Total_Sales', ascending=False)
        
        # Get top genres
        genre_total_tdf = genre_tdf.reset_index().sort_values('Total_Sales', ascending=False)
        
        # Define top genres (those making up >3% of total sales)
        genre_tops = list(genre_total_tdf.loc[genre_total_tdf['Total_Sales'] > genre_total_tdf['Total_Sales'].sum() * 0.03, 'Genre'])
        
        return {
            "genre_data": genre_total_tdf.to_dict(orient='records'),
            "genre_tops": genre_tops,
            "genre_by_region": genre_tdf.reset_index().to_dict(orient='records'),
            "genre_recent": genre_last_tdf.reset_index().to_dict(orient='records')
        }
    
    def get_geographic_data(self):
        """Get geographic sales data by region"""
        # Define regions
        regions = ['NA', 'JP', 'EU', 'Other']
        region_sales_sufix = '_Sales'
        regions_agg = {}
        
        for region in regions:
            regions_agg[region + region_sales_sufix] = 'sum'
            
        regions_agg['Total_Sales'] = 'sum'
        
        # Geographic analysis by year
        geo_tdf = self.tdf.groupby(['Year_of_Release']).agg(regions_agg).reset_index()
        geo_tdf = geo_tdf.sort_values('Year_of_Release', ascending=True)
        
        return {
            "geo_data": geo_tdf.to_dict(orient='records'),
            "regions": regions
        }
    
    def get_publisher_data(self):
        """Get publisher data for analysis"""
        # Replace NaN values in Publisher
        pub_tdf = self.tdf.copy()
        
        # Group by publisher
        pub_tdf = pub_tdf.groupby(['Publisher']).agg({'Total_Sales': ['sum', 'count']}).reset_index()
        pub_tdf.columns = ['Publisher', 'Sales_Sum', 'Sales_Count']
        
        # Remove Unknown publishers
        pub_tdf = pub_tdf[pub_tdf['Publisher'] != 'Unknown']
        
        # Filter publishers (those with >1% of sales or >1% of games)
        pub_tdf = pub_tdf[(pub_tdf['Sales_Sum'] > pub_tdf['Sales_Sum'].sum() * 0.01) |
                          (pub_tdf['Sales_Count'] > pub_tdf['Sales_Count'].sum() * 0.01)]
        
        # Define top publishers
        top_publishers = [
            'Nintendo', 
            'Sony Computer Entertainment',
            'Microsoft Game Studios',
            'Konami Digital Entertainment',
            'Electronic Arts'
        ]
        
        # Publisher analysis by region
        regions = ['NA', 'JP', 'EU', 'Other']
        region_sales_sufix = '_Sales'
        regions_agg = {}
        
        for region in regions:
            regions_agg[region + region_sales_sufix] = 'sum'
            
        regions_agg['Total_Sales'] = 'sum'
        
        # Publisher by genre analysis
        pub_genre_df = self.tdf.groupby(['Publisher', 'Genre']).agg(regions_agg).reset_index()
        pub_genre_df = pub_genre_df[pub_genre_df['Publisher'].isin(top_publishers)]
        
        # Get top genres
        genre_total_tdf = self.tdf.groupby(['Genre']).agg({'Total_Sales': 'sum'}).reset_index()
        genre_total_tdf = genre_total_tdf.sort_values('Total_Sales', ascending=False)
        genre_tops = list(genre_total_tdf.loc[genre_total_tdf['Total_Sales'] > genre_total_tdf['Total_Sales'].sum() * 0.03, 'Genre'])
        
        pub_genre_df = pub_genre_df[pub_genre_df['Genre'].isin(genre_tops)]
        
        return {
            "publishers": pub_tdf.to_dict(orient='records'),
            "top_publishers": top_publishers,
            "pub_genre": pub_genre_df.to_dict(orient='records')
        }
    
    def get_platform_counts(self):
        """Get platform count data over years"""
        top_tdf = self.tdf.groupby(['Platform', 'Year_of_Release']).agg({'Total_Sales': 'count'}).reset_index()
        top_tdf.columns = ['Platform', 'Year_of_Release', 'Count']
        top_tdf = top_tdf[top_tdf['Year_of_Release'].isin([2016, 2017, 2018, 2019])]
        top_tdf = top_tdf[top_tdf['Count'] > top_tdf['Count'].sum() * 0.01]
        top_tdf['Year_of_Release'] = top_tdf['Year_of_Release'].astype(str)
        
        return top_tdf.to_dict(orient='records')
    
    def get_rating_data(self):
        """Get ESRB rating data"""
        if 'Rating' in self.tdf.columns:
            esrb_tdf = self.tdf.groupby('Rating').agg({'Total_Sales': 'sum'}).reset_index()
            
            # Get top ratings
            esrb_tops = list(esrb_tdf.loc[esrb_tdf['Total_Sales'] > esrb_tdf['Total_Sales'].sum() * 0.03, 'Rating'])
            
            # Rating by region
            regions = ['NA', 'JP', 'EU', 'Other']
            region_sales_sufix = '_Sales'
            regions_agg = {}
            
            for region in regions:
                regions_agg[region + region_sales_sufix] = 'sum'
                
            regions_agg['Total_Sales'] = 'sum'
            
            esbr_region_tdf = self.tdf[self.tdf['Rating'].isin(esrb_tops)].groupby(['Rating']).agg(regions_agg)
            
            # Rating by genre
            genre_total_tdf = self.tdf.groupby(['Genre']).agg({'Total_Sales': 'sum'}).reset_index()
            genre_total_tdf = genre_total_tdf.sort_values('Total_Sales', ascending=False)
            genre_tops = list(genre_total_tdf.loc[genre_total_tdf['Total_Sales'] > genre_total_tdf['Total_Sales'].sum() * 0.03, 'Genre'])
            
            esrb_genre_tdf = self.tdf[self.tdf['Rating'].isin(esrb_tops)].groupby(['Rating', 'Genre']).agg({'Total_Sales': 'sum'}).reset_index()
            esrb_genre_tdf = esrb_genre_tdf[esrb_genre_tdf['Genre'].isin(genre_tops)]
            
            return {
                "ratings": esrb_tdf.to_dict(orient='records'),
                "top_ratings": esrb_tops,
                "rating_by_region": esbr_region_tdf.reset_index().to_dict(orient='records'),
                "rating_by_genre": esrb_genre_tdf.to_dict(orient='records')
            }
        else:
            return {"message": "Rating data not available in this dataset"}
    
    def get_sunburst_data(self):
        """Get data for sunburst visualizations"""
        # Get top genres
        genre_total_tdf = self.tdf.groupby(['Genre']).agg({'Total_Sales': 'sum'}).reset_index()
        genre_total_tdf = genre_total_tdf.sort_values('Total_Sales', ascending=False)
        genre_tops = list(genre_total_tdf.loc[genre_total_tdf['Total_Sales'] > genre_total_tdf['Total_Sales'].sum() * 0.03, 'Genre'])
        
        # Get top platforms
        platform_tdf = self.tdf.groupby(['Platform', 'Year_of_Release']).agg({'Total_Sales': 'sum'}).reset_index()
        platform_tdf = platform_tdf.sort_values('Year_of_Release', ascending=True)
        platform_sum_tdf = platform_tdf.groupby(['Platform']).agg({'Total_Sales': 'sum'}).reset_index()
        platform_sum_tdf = platform_sum_tdf[platform_sum_tdf['Total_Sales'] > platform_sum_tdf['Total_Sales'].sum() * 0.03]
        platform_tops = list(platform_sum_tdf['Platform'])[:4]
        
        # Get top publishers
        top_publishers = [
            'Nintendo', 
            'Sony Computer Entertainment',
            'Microsoft Game Studios',
            'Konami Digital Entertainment',
            'Electronic Arts'
        ]
        
        # Create dataframes for sunburst visualizations
        plat_genre_df = self.tdf[(self.tdf['Genre'].isin(genre_tops[:4])) & (self.tdf['Platform'].isin(platform_tops[:4]))]
        genre_pub_df = self.tdf[(self.tdf['Genre'].isin(genre_tops[:4])) & (self.tdf['Publisher'].isin(top_publishers[:5]))]
        plat_pub_df = self.tdf[(self.tdf['Platform'].isin(platform_tops[:4])) & (self.tdf['Publisher'].isin(top_publishers[:5]))]
        
        # Create all together dataframe
        genre_pub_plat_df = self.tdf[(self.tdf['Genre'].isin(genre_tops[:4])) & 
                                 (self.tdf['Publisher'].isin(top_publishers[:5])) & 
                                 (self.tdf['Platform'].isin(platform_tops[:4]))]
        
        return {
            "platform_genre": plat_genre_df[['Genre', 'Platform', 'Total_Sales']].to_dict(orient='records'),
            "genre_publisher": genre_pub_df[['Genre', 'Publisher', 'Total_Sales']].to_dict(orient='records'),
            "platform_publisher": plat_pub_df[['Platform', 'Publisher', 'Total_Sales']].to_dict(orient='records'),
            "all_together": genre_pub_plat_df[['Genre', 'Platform', 'Publisher', 'Total_Sales']].to_dict(orient='records')
        }

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