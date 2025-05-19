import requests
from bs4 import BeautifulSoup
import json

def scrape_team_rankings(urls, sport_key):
    """
    Scrape TeamRankings URLs from multiple pages for a specific sport
    
    Args:
        urls (list): List of URLs to scrape
        sport_key (str): Key to use for this sport in the results dictionary
    
    Returns:
        list: List of all scraped URLs for this sport
    """
    all_sport_urls = []
    
    for url in urls:
        try:
            # Send a GET request to the URL
            response = requests.get(url)
            
            # Check if the request was successful
            if response.status_code != 200:
                print(f"Failed to fetch {url}. Status code: {response.status_code}")
                continue
            
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all links within the expand-content sections
            all_links = soup.select('.expand-content a')
            
            # Extract URLs from all links and add to the list
            page_urls = ["https://www.teamrankings.com" + link['href'] for link in all_links]
            all_sport_urls.extend(page_urls)
            
            print(f"Scraped {len(page_urls)} URLs from {url}")
            
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
    
    print(f"Total URLs for {sport_key}: {len(all_sport_urls)}")
    return all_sport_urls

def main():
    # Define URLs to scrape for each sport
    # You can modify this dictionary to add/remove sports and URLs
    sports_urls = {
        'CBB': [
            "https://www.teamrankings.com/ncaa-basketball/stat/points-per-game",
            "https://www.teamrankings.com/ncaa-basketball/ranking/predictive-by-other/",
            "https://www.teamrankings.com/ncb/rpi/"
        ],
        'NFL': [
            "https://www.teamrankings.com/nfl/stat/points-per-game",
            "https://www.teamrankings.com/nfl/ranking/predictive-by-other/"
        ],
        'NBA': [
            "https://www.teamrankings.com/nba/stat/points-per-game",
            "https://www.teamrankings.com/nba/ranking/predictive-by-other/"
        ],
        'CFB': [
            "https://www.teamrankings.com/college-football/stat/points-per-game",
            "https://www.teamrankings.com/college-football/ranking/predictive-by-other"
        ],
        'MLB': [
            "https://www.teamrankings.com/mlb/stat/runs-per-game",
            "https://www.teamrankings.com/mlb/ranking/predictive-by-other/"
        ],
    }
    
    # Dictionary to store results
    results = {}
    
    # Process each sport
    for sport, urls in sports_urls.items():
        results[sport] = scrape_team_rankings(urls, sport)
    
    # Save to file
    with open('team_rankings_all_sports_urls.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Data saved to team_rankings_all_sports_urls.json")

if __name__ == "__main__":
    main()