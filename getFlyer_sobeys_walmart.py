# Name: Dong Han
# Mail: dongh@mun.ca
import time,os
from datetime import datetime
import pandas as pd
from selenium import webdriver
from selenium.common import TimeoutException, NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
import random
from urllib.parse import urlparse

def scroll_within_iframe(driver):
    screen_height = driver.execute_script("return window.screen.height;")
    i = 1
    while True:
        driver.execute_script(f"window.scrollTo(0, {screen_height * i});")
        i += 1
        time.sleep(2)  # Adjust time as needed
        scroll_height = driver.execute_script("return document.body.scrollHeight;")
        if (screen_height * i) > scroll_height:
            break

def getGroceryShopName(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc.split(".")[-2]

def saveFile(groceryShop,csvFileName,all_items):
    folder_path = 'scraped_draft_data'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    csvFilePath = os.path.join(folder_path, csvFileName)

    # Save data to CSV after each page
    pd.DataFrame(all_items).to_csv(csvFilePath, index=False)
    print(f"the {groceryShop} flyer data is Scraped and Saved as '{csvFileName}' in folder '{folder_path}'")
    return csvFilePath

def getFlyer(url):
    today_date = datetime.now().strftime("%Y-%m-%d")
    groceryShop = getGroceryShopName(url)

    # Set up Chrome options for Selenium
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36")
    options.add_argument("--headless")

    groceryData = {
        "sobeys":
            {
                "csvFileName": "sobeysFlyer"+"_"+today_date+".csv",
                "iframeParentSelector":"#circ_div > main",
                "driver": webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
            },
        "walmart":
            {
                "csvFileName": "walmartFlyer" + "_" + today_date + ".csv",
                "iframeParentSelector": "#flipp-flyer2-container > main",
                "driver": webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()),options=options)
            }
    }

    # get Selenium
    driver = groceryData[groceryShop]["driver"]

    # List all items from the flyer
    all_items = []

    driver.get(url)

    # Random delay
    time.sleep(random.uniform(1, 3))
    # Simulate human-like mouse movements
    action = ActionChains(driver)
    action.move_to_element(driver.find_element(By.TAG_NAME, 'body')).perform()
    action.move_by_offset(random.randint(10, 100), random.randint(10, 100)).perform()

    # Wait for the page to load completely
    wait = WebDriverWait(driver, 30)
    wait.until(lambda d: d.execute_script("return document.readyState === 'complete';"))

    # Attempt to find the iframe
    try:
        # Locate the parent element <DONT try to locate the iframe directly>
        parent_element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, groceryData[groceryShop]["iframeParentSelector"])))

        # Find the iframe within the parent element
        iframe = parent_element.find_element(By.TAG_NAME, "iframe")
        driver.switch_to.frame(iframe)

        # Scroll within the iframe (Have to cuz the flyer images are loaded inside the iframe)
        scroll_within_iframe(driver)

        # Locate the parent elements
        parent_elements = driver.find_elements(By.CSS_SELECTOR, "body > flipp-router > flipp-publication-page > div > div.sfml-wrapper")

        for parent in parent_elements:
            # Locate the child elements within each parent
            child_elements = parent.find_elements(By.TAG_NAME, "sfml-flyer-image-a")


            for child in child_elements:
                item_data = {
                    "aria-label": child.get_attribute("aria-label"),
                    "item-type": child.get_attribute("item-type"),
                    "item-id": child.get_attribute("item-id"),
                    "img-link": child.get_attribute("href"),
                    "src-rect": child.get_attribute("src-rect"),
                    "item-type-number": child.get_attribute("item-type-number")
                }
                all_items.append(item_data)

        csvFileName = groceryData[groceryShop]["csvFileName"]
        csvFilePath = saveFile(groceryShop, csvFileName, all_items)
        return csvFilePath,groceryShop

        '''
        # this can be used to locate the button which contains "data-product-id"
        product_elements = driver.find_elements(By.XPATH, "//*[@data-product-id]")
        for element in product_elements:
            item_data = {
                "aria-label": element.get_attribute("aria-label"),
                "item-type": element.get_attribute("item-type"),
                "item-id": element.get_attribute("item-id"),
                "img-link": element.get_attribute("href"),
                "src-rect": element.get_attribute("src-rect"),
                "item-type-number": element.get_attribute("item-type-number")
            }
            all_items.append(item_data)
        return all_items
        '''

    except TimeoutException:
        print("Timed out waiting for page to load")
        driver.save_screenshot('debug_screenshot_after_timeout.png')
        return -1,-1
    except NoSuchElementException:
        print("Could not find the iframe or elements within it.")
        return -1,-1
    finally:
        driver.quit()

    '''
    # Could use below way but notice that it will return a typical Selenium WebElement representation
    #[<selenium.webdriver.remote.webelement.WebElement (session="7ae7b5b7a27ea372601d9e42c1d7867c", element="45A1B804CE5B7BCA8DA6F5053FD7A15D_element_304")>]
    # So next should use '.text' to get its text content, '.get_attribute()' to retrieve specific attributes
    sfml_wrapper  = wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'sfml-wrapper')))
    try:
        flyer_wrapper = driver.find_elements(By.CLASS_NAME, "sfml-wrapper")
        driver.save_screenshot('debug_screenshot.png')
    except 
    ......
    '''

def main():
    sobeys_url = "https://www.sobeys.com/en/flyer/"
    sobeysItemSavedFile,shopName = getFlyer(sobeys_url)

    # walmart_url = "https://www.walmart.ca/en/flyer"
    # walmartFlyer,shopName = getFlyer(walmart_url)

if __name__ == '__main__':
    main()
