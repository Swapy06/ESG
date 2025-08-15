import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import yfinance as yf

# Initialize NLTK sentiment analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

class ESGScorer:
    def __init__(self):
        self.company_data = {}
        self.esg_weights = {
            'environmental': 0.4,
            'social': 0.3,
            'governance': 0.3
        }
        
    def fetch_yfinance_esg(self, ticker):
        """Fetch ESG data from Yahoo Finance"""
        try:
            stock = yf.Ticker(ticker)
            esg_data = stock.sustainability
            
            if esg_data is not None:
                self.company_data[ticker] = {
                    'environmental': esg_data.get('environmentScore', np.nan),
                    'social': esg_data.get('socialScore', np.nan),
                    'governance': esg_data.get('governanceScore', np.nan),
                    'total': esg_data.get('totalEsg', np.nan)
                }
            else:
                print(f"No ESG data available for {ticker}")
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
    
    def scrape_sustainability_report(self, url):
        """Scrape and analyze sustainability reports"""
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            text = ' '.join([p.get_text() for p in soup.find_all('p')])
            
            # Analyze sentiment and keyword frequency
            sentiment = sia.polarity_scores(text)
            keywords = {
                'environmental': ['carbon', 'emission', 'renewable', 'waste', 'energy'],
                'social': ['diversity', 'community', 'human rights', 'employee', 'safety'],
                'governance': ['ethics', 'board', 'compensation', 'transparency', 'anti-corruption']
            }
            
            keyword_counts = {category: sum(text.lower().count(word) for word in words) 
                            for category, words in keywords.items()}
            
            return {
                'sentiment': sentiment['compound'],
                'keyword_counts': keyword_counts
            }
        except Exception as e:
            print(f"Error scraping report: {str(e)}")
            return None
    
    def calculate_custom_score(self, ticker, report_url=None):
        """Calculate custom ESG score combining multiple sources"""
        if ticker not in self.company_data:
            self.fetch_yfinance_esg(ticker)
            
        base_scores = self.company_data.get(ticker, {})
        report_analysis = self.scrape_sustainability_report(report_url) if report_url else None
        
        # Calculate component scores (0-100 scale)
        scores = {}
        for category in self.esg_weights.keys():
            # Start with Yahoo Finance score if available
            score = base_scores.get(category, 50)  # Default to 50 if no data
            
            # Adjust based on report analysis
            if report_analysis:
                # Boost score based on positive sentiment
                score += report_analysis['sentiment'] * 10
                
                # Boost based on keyword frequency (cap at +20)
                keyword_boost = min(report_analysis['keyword_counts'].get(category, 0) * 2, 20)
                score += keyword_boost
                
            # Ensure score stays within bounds
            scores[category] = max(0, min(100, score))
            
        # Calculate weighted total score
        total_score = sum(scores[cat] * weight for cat, weight in self.esg_weights.items())
        
        return {
            'components': scores,
            'total_score': total_score,
            'yahoo_finance_score': base_scores.get('total', None)
        }
    
    def compare_companies(self, tickers, report_urls=None):
        """Compare multiple companies' ESG performance"""
        results = {}
        report_urls = report_urls or [None] * len(tickers)
        
        for ticker, url in zip(tickers, report_urls):
            results[ticker] = self.calculate_custom_score(ticker, url)
            
        return results
    
    def visualize_results(self, comparison_results):
        """Visualize ESG comparison"""
        df = pd.DataFrame.from_dict({
            ticker: data['components'] 
            for ticker, data in comparison_results.items()
        }, orient='index')
        
        df['Total Score'] = [data['total_score'] for data in comparison_results.values()]
        
        # Plotting
        fig, ax = plt.subplots(figsize=(12, 6))
        df[['environmental', 'social', 'governance']].plot(
            kind='bar', stacked=True, ax=ax,
            color=['#2ecc71', '#3498db', '#9b59b6']
        )
        ax.plot(df.index, df['Total Score'], 'ko-', label='Total Score')
        
        ax.set_title('ESG Performance Comparison')
        ax.set_ylabel('Score (0-100)')
        ax.legend(loc='upper right')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return fig

# Example Usage
if __name__ == "__main__":
    esg = ESGScorer()
    
    # Example companies (add your own tickers)
    companies = ['AAPL', 'MSFT', 'TSLA', 'JPM']
    
    # Example report URLs (replace with actual URLs)
    reports = [
        'https://www.apple.com/environment/pdf/Apple_Environmental_Progress_Report_2023.pdf',
        'https://www.microsoft.com/en-us/corporate-responsibility/sustainability',
        'https://www.tesla.com/ns_videos/2021-tesla-impact-report.pdf',
        None  # No report for JPM
    ]
    
    # Get comparison results
    results = esg.compare_companies(companies, reports)
    
    # Display results
    for ticker, data in results.items():
        print(f"\n{ticker} ESG Analysis:")
        print(f"Custom Total Score: {data['total_score']:.1f}")
        if data['yahoo_finance_score']:
            print(f"Yahoo Finance Score: {data['yahoo_finance_score']}")
        print("Component Scores:")
        for cat, score in data['components'].items():
            print(f"  {cat.capitalize()}: {score:.1f}")
    
    # Visualize comparison
    esg.visualize_results(results)