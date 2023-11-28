import requests
from bs4 import BeautifulSoup
from selenium import webdriver

driver = webdriver.Chrome()
youtube_url = "https://www.youtube.com/watch?v=X7158uQk1yI"
driver.get(youtube_url)

html = BeautifulSoup(driver.page_source,'lxml')
print(html)

# class = "ytp-heat-map-path"
# 히트맵 그래프 정보가 담긴 html태그의 클래스


'''
response = requests.get("https://www.youtube.com/watch?v=X7158uQk1yI")
soup = BeautifulSoup(response.text, 'html.parser')
print(soup)
program_names = soup.find("path", id="3")
'''



