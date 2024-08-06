from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time

# Set up the WebDriver with Chrome options
def setup_driver():
    options = Options()
    options.add_experimental_option('detach', True)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver

# Parse individual thread to get the desired information
def parse_thread(driver, thread_index):
    # Use XPath to directly access the thread link and navigate
    thread_xpath = f"(//div[@class='structItem-title']//a)[{thread_index}]"
    try:
        thread_link = driver.find_element(By.XPATH, thread_xpath).get_attribute('href')
        driver.get(thread_link)
        time.sleep(2)  # Wait for the page to load
    except Exception as e:
        print(f"Failed to access thread at index {thread_index}: {e}")
        return None, None, []

    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Extract thread details
    title = soup.select_one('h1.p-title-value').text.strip() if soup.select_one('h1.p-title-value') else "No Title"
    content = soup.select_one('div.p-body-content').text.strip() if soup.select_one('div.p-body-content') else "No Content"

    # Extract user comments
    comments = []
    for message in soup.select('article'):
        user = message.select_one('h4 a').text.strip() if message.select_one('h4 a') else "Unknown User"
        message_content = message.select_one('div.message-body').get_text(strip=True) if message.select_one('div.message-body') else "No Message"
        comments.append((user, message_content))

    return title, content, comments

# Crawl the forum pages and gather data
def crawl_forum(base_url, num_pages=2):
    driver = setup_driver()
    all_threads = []

    try:
        for page in range(1, num_pages + 1):
            page_url = f"{base_url}?page={page}"
            driver.get(page_url)
            time.sleep(2)  # Wait for the page to load

            # Estimate the number of threads on a page, typically you can set a reasonable range
            for thread_index in range(1, 21):  # Adjust the range according to actual number of threads visible
                title, content, comments = parse_thread(driver, thread_index)
                if title and content:
                    for user, text in comments:
                        all_threads.append({
                            'Thread Title': title,
                            'Thread Content': content,
                            'User': user,
                            'Message': text
                        })

    finally:
        driver.quit()

    df = pd.DataFrame(all_threads)
    df.to_csv('forum_threads.csv', index=False)

# Start crawling
base_url = "https://www.dentistry-forums.com/forums/patient-forum.17/"
crawl_forum(base_url, num_pages=2)
