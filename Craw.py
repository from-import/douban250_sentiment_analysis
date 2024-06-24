import time
import requests
from bs4 import BeautifulSoup

# 爬取豆瓣电影Top250中第n个电影的前m条影评
def crawl_douban_movie_reviews(n, m):
    base_url = 'https://movie.douban.com/top250?start='
    data = []

    movie_count = 0
    review_count = 0

    # 豆瓣Top250电影每页有25部电影，总共10页
    for i in range(0, 10):
        if movie_count >= n:
            break  # 如果已经爬取到第n个电影，就停止爬取

        url = base_url + str(i * 25)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://www.google.com/',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'TE': 'Trailers',
            'Pragma': 'no-cache',
            'Cache-Control': 'no-cache'
        }

        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'lxml')

        # 找到当前页面的所有电影详情链接
        movie_links = soup.find_all('div', {'class': 'hd'})

        # 逐一访问电影详情页面并爬取评论
        for movie_link in movie_links:
            movie_count += 1

            if movie_count == n:  # 如果是第n个电影，就开始爬取评论
                link = movie_link.a['href']
                movie_page_response = requests.get(link, headers=headers)  # 添加headers
                movie_page_soup = BeautifulSoup(movie_page_response.text, 'lxml')

                # 获取电影名称
                movie_name_element = movie_page_soup.find('span', property='v:itemreviewed')
                movie_name = movie_name_element.text if movie_name_element else None  # 如果找不到元素，movie_name为None

                print(f"爬取第 {n} 部电影: {movie_name}")

                # 获取电影短评
                movie_short_reviews = movie_page_soup.find_all('div', attrs={'class': 'comment'})

                for movie_short_review in movie_short_reviews:
                    if review_count >= m:
                        break  # 如果已经爬取到前m条评论，就停止爬取

                    review = movie_short_review.find('span', attrs={'class': 'short'}).text
                    data.append((movie_name, review))
                    review_count += 1
                    print(f"爬取评论 {review_count}: {review}")

                time.sleep(2)  # 每爬取一部电影就暂停2秒，防止频繁请求导致被封IP
                break

    return data
