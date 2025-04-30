import os
import time
import requests
from PIL import Image
from io import BytesIO
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class CrawlerGoogleImages:
    def __init__(self, keyword):
        self.keyword = keyword
        self.url = f"https://www.google.com/search?q={keyword}&tbm=isch"

    def init_browser(self):
        chrome_options = webdriver.ChromeOptions()

        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--lang=en-US")
        chrome_options.add_argument("--headless")  # 无头模式，隐藏浏览器窗口
        chrome_options.add_argument("--disable-gpu")  # 禁用GPU加速，适用于某些环境下无头模式
        chrome_options.add_argument("--window-size=1920x1080")  # 设置窗口大小，避免无头模式下页面渲染问题
        chrome_options.add_argument("--no-sandbox")  # 适用于Linux系统，防止无权限问题
        chrome_options.add_argument("--disable-dev-shm-usage")  # 避免/dev/shm共享内存空间不足的问题

        chrome_options.add_argument("--remote-debugging-port=1255")  # 远程调试端口
        browser = webdriver.Chrome(options=chrome_options)

        browser.get(self.url)
        browser.maximize_window()
        return browser

    def download_images(self, browser, num_images=5):
        img_url_dic = []
        img_list = []
        pos = 0
        max_attempts = 200  # 最大检索次数
        attempts = 0  # 当前尝试次数

        # Scroll and find images until the desired number of high-resolution images is collected or max_attempts is reached
        while len(img_list) < num_images and attempts < max_attempts:
            pos += 500
            js = 'var q=document.documentElement.scrollTop=' + str(pos)
            browser.execute_script(js)
            time.sleep(1)

            WebDriverWait(browser, 10).until(
                EC.presence_of_all_elements_located((By.TAG_NAME, 'img'))
            )

            try:
                img_elements = browser.find_elements(By.TAG_NAME, 'img')
                session = requests.Session()
                session.trust_env = False

                # 如果梯子是 HTTP 代理，可以这样配置
                # 但要注意，如果梯子不支持 HTTPS，就不要在这里配置 'https' 键
                proxies = {
                    "http": "http://127.0.0.1:1080",  # 你的HTTP代理
                    "https": "http://127.0.0.1:1080",  # 不支持HTTPS就先不填这个
                }
                session.proxies.update(proxies)
                for img_element in img_elements:
                    if len(img_list) >= num_images:
                        break

                    # Try to get high-resolution image URL
                    img_url = None
                    srcset = img_element.get_attribute('srcset')
                    if srcset:
                        img_urls = [url.split(' ')[0] for url in srcset.split(',')]
                        img_url = img_urls[-1]  # Last one is usually the highest resolution
                    else:
                        img_url = img_element.get_attribute('data-src') or img_element.get_attribute('src')

                    if img_url and isinstance(img_url, str) and 'images' in img_url:
                        if img_url not in img_url_dic:
                            img_url_dic.append(img_url)
                            # print(f"Found image URL: {img_url}")

                            session = requests.Session()
                            session.trust_env = False
                            proxies = {
                                "http": "http://127.0.0.1:1080",  # 你的HTTP代理
                                "https": "http://127.0.0.1:1080",  # 不支持HTTPS就先不填这个
                            }
                            session.proxies.update(proxies)
                            r = session.get(url=img_url)

                            try:
                                img = Image.open(BytesIO(r.content))
                                width, height = img.size
                                if width > 50 and height > 50:  # Ensure it's a high-resolution image
                                    img_list.append(img)  # Add the image object to the list
                                    if len(img_list) >= num_images:
                                        break
                                else:
                                    pass
                                    # print(f"Image is not high-resolution, skipping.")
                            except Exception as e:
                                print(f"Failed to process image: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")

            attempts += 1  # Increment the attempts counter

        if attempts >= max_attempts:
            print("Reached maximum number of attempts. Returning collected images.")

        return img_list  # Return the list of image objects

    def run(self, num_images=5):
        browser = self.init_browser()
        img_list = self.download_images(browser, num_images)
        browser.close()
        return img_list  # Return the list of images
# Encapsulate the entire process in a function
def get_image_by_keyword(keyword):
    crawler = CrawlerGoogleImages(keyword)
    image = crawler.run()
    return image

# Example usage:
if __name__ == '__main__':
    key_word = 'sodium_chloride'  # Example keyword
    image = get_image_by_keyword(key_word)

    if image:
        # Save the image locally or process it as needed
        image.show()  # This will display the image
    else:
        print("No image found.")
