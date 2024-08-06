from selenium.webdriver.chrome.service import Service
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import json
import time
from webdriver_manager.chrome import ChromeDriverManager

# Set up Chrome options
options = Options()
options.add_argument("--headless")  # Run Chrome in headless mode

# Initialize the WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

def get_forum_posts(page_url):
    driver.get(page_url)
    wait = WebDriverWait(driver, 10)
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Get the titles of the forum posts
    posts = []
    for i in range(1, 51):  # Loop through the first 50 posts on the page
        xpath = f"/html/body/div[1]/div[4]/div/div[2]/div[3]/div/div/div[2]/div/div/div/div[{i}]/div[2]/div[1]/a"
        try:
            post_element = wait.until(EC.presence_of_element_located((By.XPATH, xpath)))
            posts.append(post_element.get_attribute('href'))
        except Exception as e:
            print(f"Error fetching post link at index {i}: {e}")
            break

    return posts

def get_post_content(post_url, post_id):
    driver.get(post_url)
    wait = WebDriverWait(driver, 10)
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Get the post title from <div class="p-title"> and <h1 class="p-title-value">
    title_div = soup.find('div', class_='p-title')
    if title_div:
        title_element = title_div.find('h1', class_='p-title-value')
        title = title_element.text.strip() if title_element else "No Title Found"
    else:
        print(f"Title div not found on {post_url}")
        title = "No Title Found"

    # Collect all replies
    replies = []
    articles = soup.find_all('article', {'class': 'message'})
    if not articles:
        print(f"No articles found on {post_url}")

    for article in articles:
        username_element = article.find('h4', class_='message-name')
        text_content_element = article.find('div', class_='bbWrapper')
        if username_element and text_content_element:
            username = username_element.text.strip()
            text_content = text_content_element.text.strip()
            replies.append(f"{username}: {text_content}")
        else:
            print(f"Article missing elements on {post_url}")

    return {
        'id': post_id,
        'title': title,
        'dialogue': ' '.join(replies)
    }

def scrape_forum(start_url):
    all_data = []
    post_id = 1
    for page_number in range(1, 45):  # Adjust the range according to the number of pages
        page_url = f"{start_url}?page={page_number}"
        print(f"Scraping page: {page_number}")
        post_urls = get_forum_posts(page_url)
        for post_url in post_urls:
            print(f"Scraping post: {post_url}")
            post_data = get_post_content(post_url, post_id)
            all_data.append(post_data)
            post_id += 1
            time.sleep(1)  # To avoid being blocked by the server
    return all_data

# Start scraping
start_url = 'https://www.dentistry-forums.com/forums/patient-forum.17/'
data = scrape_forum(start_url)

# Output the data to a JSON file
with open('forum_data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# Close the WebDriver
driver.quit()
