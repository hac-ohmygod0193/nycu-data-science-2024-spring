from queue import Queue
import threading
import requests
from bs4 import BeautifulSoup
import time
import json
import argparse
from tqdm import tqdm
from collections import defaultdict
import re
import os
import concurrent.futures
from functools import partial
from urllib.parse import urlparse


def fetch_page(url: str):
    """Fetch the page content"""
    headers = {
        "Cookie": "over18=1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    }
    requests.adapters.DEFAULT_RETRIES = 5
    response = requests.get(url, headers=headers)
    try:
        while response.status_code != 200:
            time.sleep(5)
            response = requests.get(url, headers=headers)
    except:
        pass
    return response.text


def convert_date_format(date_str):
    date = date_str.replace("/", "").replace(" ", "")
    if len(date) == 3:
        date = "0" + date
    return date


def extract_article_info(tag, filter_text):
    """Extract article information from a single div tag"""
    title = tag.find(name="div", attrs={"class": "title"})
    count = tag.find(name="div", attrs={"class": "nrec"})
    a_tag = title.find("a")
    try:
        link = f'https://www.ptt.cc{a_tag["href"]}'
        date = tag.find(name="div", attrs={"class": "date"})
        if filter_text == date.text or "[公告]" in title.text or title.text is None:
            return
        article = {
            "date": convert_date_format(date.text),
            "title": title.text[1:-1],
            "url": link,
        }
    except:
        pass
    with open("articles.jsonl", "a", encoding="utf-8") as outfile:
        outfile.write(json.dumps(article, ensure_ascii=False) + "\n")
    if count.text == "爆":
        with open("popular_articles.jsonl", "a", encoding="utf-8") as outfile:
            outfile.write(json.dumps(article, ensure_ascii=False) + "\n")


def crawl():

    start_index = 3400
    end_index = 4000
    step = 1
    start = 0
    end = -1
    current_date = ""
    with open("articles.jsonl", "w", encoding="utf-8") as outfile:
        outfile.write("")
    with open("popular_articles.jsonl", "w", encoding="utf-8") as outfile:
        outfile.write("")
    for i in tqdm(range(start_index, end_index + 1, step)):
        # for i in range(start_index, end_index + 1, step):
        time.sleep(0.01)
        url = f"https://www.ptt.cc/bbs/Beauty/index{i}.html"
        html = fetch_page(url)
        soup = BeautifulSoup(html, "html.parser")
        tags = soup.find_all(name="div", attrs={"class": "r-ent"})
        if start == 0:
            for tag in tags:
                date = tag.find(name="div", attrs={"class": "date"})
                if " 1/01" == date.text:
                    start = 1
                    break
            if start == 0:
                continue
        filter_text = ""
        if current_date == "" or current_date == " 1/01":
            filter_text = "12/31"
        else:
            filter_text = " 1/01"
        for tag in tags:
            date = tag.find(name="div", attrs={"class": "date"})
            if current_date == " 1/01":
                end = 0
            if current_date == "12/31" and date.text == " 1/01":
                if end == 0:
                    end = 1
                    break
            current_date = date.text
            extract_article_info(tag, filter_text)
        if end == 1:
            break


def is_date_between(date, start_date, end_date):
    """
    Check if a date is between start date and end date.
    Args:
        date (str): The date to examine in the format MMDD.
        start_date (str): The start date in the format MMDD.
        end_date (str): The end date in the format MMDD.

    Returns:
        bool: True if the date is between start date and end date, False otherwise.
    """
    # Convert dates to a standardized format (MM/DD)
    date = "{:0>2}/{:0>2}".format(date[:2], date[2:])
    start_date = "{:0>2}/{:0>2}".format(start_date[:2], start_date[2:])
    end_date = "{:0>2}/{:0>2}".format(end_date[:2], end_date[2:])

    # Check if the date is between start date and end date
    return start_date <= date <= end_date


def binary_search_first_date(articles, target_date):
    """
    Perform binary search to find the index of the first article with a date >= target_date.

    Args:
        articles (list): List of articles sorted by date.
        target_date (str): The target date to find.

    Returns:
        int: Index of the first article with a date >= target_date.
    """
    left, right = 0, len(articles) - 1
    while left <= right:
        mid = (left + right) // 2
        if articles[mid]["date"] >= target_date:
            if mid == 0 or articles[mid - 1]["date"] < target_date:
                return mid
            right = mid - 1
        else:
            left = mid + 1
    return -1


def binary_search_last_date(articles, target_date):
    """
    Perform binary search to find the index of the last article with a date <= target_date.

    Args:
        articles (list): List of articles sorted by date.
        target_date (str): The target date to find.

    Returns:
        int: Index of the last article with a date <= target_date.
    """
    left, right = 0, len(articles) - 1
    while left <= right:
        mid = (left + right) // 2
        if articles[mid]["date"] <= target_date:
            if mid == len(articles) - 1 or articles[mid + 1]["date"] > target_date:
                return mid
            left = mid + 1
        else:
            right = mid - 1
    return -1


def push_save_to_json(data, start_date, end_date):
    filename = f"push_{start_date}_{end_date}.json"
    with open(filename, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)
    print(f"Data saved to {filename}")


def push_worker(articles_queue, start_date, end_date, push_counts, boo_counts):
    while True:
        article = articles_queue.get()
        if article is None:
            break

        date = article["date"]
        title = article["title"]
        url = article["url"]
        if is_date_between(date, start_date, end_date):
            html = fetch_page(url)
            soup = BeautifulSoup(html, "html.parser")
            tags = soup.find_all(name="div", attrs={"class": "push"})
            for tag in tags:
                push_tag = tag.find(name="span", attrs={"class": "hl push-tag"})
                if push_tag is None:
                    push_tag = tag.find(name="span", attrs={"class": "f1 hl push-tag"})
                user_id = tag.find(
                    name="span", attrs={"class": "f3 hl push-userid"}
                ).text
                if push_tag.text == "推 ":
                    push_counts[user_id] += 1
                elif push_tag.text == "噓 ":
                    boo_counts[user_id] += 1

        articles_queue.task_done()


def push(start_date, end_date):
    push_counts = defaultdict(int)
    boo_counts = defaultdict(int)
    articles = []
    with open("articles.jsonl", "r", encoding="utf-8") as infile:
        for line in infile:
            article = json.loads(line)
            articles.append(article)

    start_index = binary_search_first_date(articles, start_date)
    end_index = binary_search_last_date(articles, end_date) + 1

    articles_queue = Queue()
    for article in tqdm(
        articles[start_index:end_index], desc="Enqueuing articles", unit="article"
    ):
        articles_queue.put(article)

    num_threads = 10
    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(
            target=push_worker,
            args=(articles_queue, start_date, end_date, push_counts, boo_counts),
        )
        thread.start()
        threads.append(thread)

    total_articles = articles_queue.qsize()
    pbar = tqdm(total=total_articles, desc="Processing articles", unit="article")
    while total_articles > 0:
        articles_queue.join()
        completed_articles = total_articles - articles_queue.qsize()
        pbar.update(completed_articles)
        total_articles = articles_queue.qsize()

    pbar.close()

    for _ in range(num_threads):
        articles_queue.put(None)

    for thread in threads:
        thread.join()

    sorted_push_users = sorted(push_counts.items(), key=lambda x: (x[1],x[0]),reverse=True)[:10]
    sorted_boo_users = sorted(boo_counts.items(), key=lambda x: (x[1],x[0]),reverse=True)[:10]

    data = {
        "push": {
            "total": sum(push_counts.values()),
            "top10": [
                {"user_id": user_id, "count": count}
                for user_id, count in sorted_push_users
            ],
        },
        "boo": {
            "total": sum(boo_counts.values()),
            "top10": [
                {"user_id": user_id, "count": count}
                for user_id, count in sorted_boo_users
            ],
        },
    }
    push_save_to_json(data, start_date, end_date)


def filter_image_urls(urls):
    # Define the regex pattern to match image URLs
    pattern = (
        r'https?://(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,7}/[^"\s]*\.(?:jpg|jpeg|png|gif)'
    )

    # Filter URLs that match the pattern
    image_urls = [url for url in urls if re.match(pattern, url)]

    return image_urls


def popular_save_to_json(number_of_popular_articles, image_urls, start_date, end_date):
    filename = f"popular_{start_date}_{end_date}.json"
    data = {
        "number_of_popular_articles": number_of_popular_articles,
        "image_urls": image_urls,
    }
    with open(filename, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)
    print(f"Data saved to {filename}")


def popular_worker(articles_queue, start_date, end_date, urls):
    while True:
        article = articles_queue.get()
        if article is None:
            break

        date = article["date"]
        title = article["title"]
        url = article["url"]
        if is_date_between(date, start_date, end_date):
            html = fetch_page(url)
            soup = BeautifulSoup(html, "html.parser")
            links = soup.find_all("a", {"target": "_blank"})
            for link in links:
                urls.append(link["href"])

        articles_queue.task_done()


def popular(start_date, end_date):
    articles = []
    urls = []
    with open("popular_articles.jsonl", "r", encoding="utf-8") as infile:
        for line in infile:
            article = json.loads(line)
            articles.append(article)

    start_index = binary_search_first_date(articles, start_date)
    end_index = binary_search_last_date(articles, end_date) + 1
    number_of_popular_articles = end_index - start_index

    articles_queue = Queue()
    for article in tqdm(
        articles[start_index:end_index], desc="Enqueuing articles", unit="article"
    ):
        articles_queue.put(article)

    num_threads = 10
    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(
            target=popular_worker, args=(articles_queue, start_date, end_date, urls)
        )
        thread.start()
        threads.append(thread)

    total_articles = articles_queue.qsize()
    pbar = tqdm(total=total_articles, desc="Processing articles", unit="article")
    while total_articles > 0:
        articles_queue.join()
        completed_articles = total_articles - articles_queue.qsize()
        pbar.update(completed_articles)
        total_articles = articles_queue.qsize()

    pbar.close()

    for _ in range(num_threads):
        articles_queue.put(None)

    for thread in threads:
        thread.join()

    image_urls = filter_image_urls(urls)
    popular_save_to_json(number_of_popular_articles, image_urls, start_date, end_date)


def keyword_save_to_json(image_urls, start_date, end_date, keyword):
    filename = f"keyword_{start_date}_{end_date}_{keyword}.json"
    data = {"image_urls": image_urls}
    with open(filename, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)
    print(f"Data saved to {filename}")


def keyword_worker(articles_queue, start_date, end_date, search_keyword, urls):
    while True:
        article = articles_queue.get()
        if article is None:
            break

        date = article["date"]
        title = article["title"]
        url = article["url"]
        html = fetch_page(url)
        soup = BeautifulSoup(html, "html.parser")

        start = 0
        find_keyword = 0
        find_element = 0
        # Find the two elements between which you want to search for the keyword
        all_text = soup.find_all(string=True)
        for text in all_text:
            if start == 0 and "作者" in text:
                start = 1
            if start == 1:
                if "※ 發信站: 批踢踢實業坊(ptt.cc)" in text:
                    find_element = 1
                    break
                if search_keyword in text:
                    find_keyword = 1
        if find_keyword and find_element:
            with open(search_keyword + ".txt", "a", encoding="utf-8") as f:
                f.write(title + "\n")
            links = soup.find_all("a", {"target": "_blank"})
            for link in links:
                urls.append(link["href"])

        articles_queue.task_done()


def keyword(start_date, end_date, search_keyword):
    articles = []
    urls = []
    with open("articles.jsonl", "r", encoding="utf-8") as infile:
        for line in infile:
            article = json.loads(line)
            articles.append(article)

    start_index = binary_search_first_date(articles, start_date)
    end_index = binary_search_last_date(articles, end_date) + 1

    articles_queue = Queue()
    for article in tqdm(
        articles[start_index:end_index], desc="Enqueuing articles", unit="article"
    ):
        articles_queue.put(article)

    num_threads = 10
    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(
            target=keyword_worker,
            args=(articles_queue, start_date, end_date, search_keyword, urls),
        )
        thread.start()
        threads.append(thread)

    total_articles = articles_queue.qsize()
    pbar = tqdm(total=total_articles, desc="Processing articles", unit="article")
    while total_articles > 0:
        articles_queue.join()
        completed_articles = total_articles - articles_queue.qsize()
        pbar.update(completed_articles)
        total_articles = articles_queue.qsize()

    pbar.close()

    for _ in range(num_threads):
        articles_queue.put(None)

    for thread in threads:
        thread.join()

    image_urls = filter_image_urls(urls)
    keyword_save_to_json(image_urls, start_date, end_date, search_keyword)


def extract_popular_article_info(tag, filter_text):
    """Extract article information from a single div tag"""
    title = tag.find(name="div", attrs={"class": "title"})
    count = tag.find(name="div", attrs={"class": "nrec"})
    a_tag = title.find("a")
    link = f'https://www.ptt.cc{a_tag["href"]}'
    date = tag.find(name="div", attrs={"class": "date"})
    if filter_text == date.text or "[公告]" in title.text or title.text is None:
        return
    article = {
        "date": convert_date_format(date.text),
        "title": title.text[1:-1],
        "url": link,
    }
    if count.text == "爆" or (count.text.isdigit() and int(count.text) >= 35):
        with open("images_1_articles.jsonl", "a", encoding="utf-8") as outfile:
            outfile.write(json.dumps(article, ensure_ascii=False) + "\n")
    else:
        with open("images_0_articles.jsonl", "a", encoding="utf-8") as outfile:
            outfile.write(json.dumps(article, ensure_ascii=False) + "\n")


def crawl_popular_article():
    start_index = 3650
    end_index = 4000
    step = 1
    start = 0
    end = -1
    current_date = ""
    with open("images_0_articles.jsonl", "w", encoding="utf-8") as outfile:
        outfile.write("")
    with open("images_1_articles.jsonl", "w", encoding="utf-8") as outfile:
        outfile.write("")
    for i in tqdm(range(start_index, end_index + 1, step)):
        time.sleep(0.01)
        url = f"https://www.ptt.cc/bbs/Beauty/index{i}.html"
        html = fetch_page(url)
        soup = BeautifulSoup(html, "html.parser")
        tags = soup.find_all(name="div", attrs={"class": "r-ent"})
        if start == 0:
            for tag in tags:
                date = tag.find(name="div", attrs={"class": "date"})
                if " 1/01" == date.text:
                    start = 1
                    break
            if start == 0:
                continue
        filter_text = ""
        if current_date == "" or current_date == " 1/01":
            filter_text = "12/31"
        else:
            filter_text = " 1/01"
        for tag in tags:
            date = tag.find(name="div", attrs={"class": "date"})
            if current_date == " 1/01":
                end = 0
            if current_date == "12/31" and date.text == " 1/01":
                if end == 0:
                    end = 1
                    break
            current_date = date.text
            extract_popular_article_info(tag, filter_text)
        if end == 1:
            break


def download_image(folder_path, url):
    html = fetch_page(url)
    soup = BeautifulSoup(html, "html.parser")
    start = 0
    find_keyword = 0
    # Find the two elements between which you want to search for the keyword
    img_tags = soup.find_all("img")
    for img_tag in img_tags:

        img = img_tag.get("src")
        # Extract the file name from the URL
        parsed_url = urlparse(img)
        file_name_with_extension = os.path.basename(parsed_url.path)
        file_name, file_extension = os.path.splitext(file_name_with_extension)
        # Construct the file path
        file_path = os.path.join(folder_path, file_name_with_extension)
        if file_name_with_extension.lower().endswith(".gif"):
            print(f"Skipping GIF image: {file_name_with_extension}")
            continue
        # Check if the file already exists
        if os.path.exists(file_path):
            print(
                f"File '{file_name_with_extension}' already exists. Skipping download."
            )
            continue
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        }
        time.sleep(0.001)
        response = requests.get(img, headers=headers)
        if response.status_code == 200:

            # Write content to file
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"Image downloaded successfully as '{file_name_with_extension}'")
        elif response.status_code == 429:
            print(f"Rate limit exceeded. Waiting for 5 seconds before retrying...")
            time.sleep(5)  # Wait for 5 seconds before retrying
            # Retry the request
            response = requests.get(img)
            if response.status_code == 200:
                # Process the response as before
                pass
            else:
                print(
                    f"Failed to download image from {img}: Status code {response.status_code}"
                )
        elif response.status_code == 404:
            print(
                f"Failed to download image from {img}: Status code {response.status_code}"
            )
        else:
            print(
                f"Failed to download image from {img}: Status code {response.status_code}"
            )


def download():
    popular_articles = []
    non_popular_articles = []

    popular_urls = []
    non_popular_urls = []
    articles = []
    with open("images_1_articles.jsonl", "r", encoding="utf-8") as infile:
        for line in infile:
            article = json.loads(line)
            popular_articles.append(article["title"])
            popular_urls.append(article["url"])
    with open("images_0_articles.jsonl", "r", encoding="utf-8") as infile:
        for line in infile:
            article = json.loads(line)
            if article["url"] not in popular_urls:
                non_popular_articles.append(article["title"])
                non_popular_urls.append(article["url"])
            else:
                articles.append(article["title"])
    print(len(non_popular_articles), len(popular_articles), len(articles))
    for title, url in tqdm(zip(non_popular_articles, non_popular_urls)):
        print(title)
        download_image("./images_0", url)
    for title, url in tqdm(zip(popular_articles, popular_urls)):
        print(title)
        download_image("./images_1", url)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl PTT Beauty board.")
    parser.add_argument(
        "action",
        choices=["crawl", "push", "popular", "keyword", "download"],
        help="Action to perform: crawl or push.",
    )
    parser.add_argument(
        "start_date", nargs="?", type=str, default=None, help="Start date for action."
    )
    parser.add_argument(
        "end_date", nargs="?", type=str, default=None, help="End date for action."
    )
    parser.add_argument(
        "search_keyword", nargs="?", type=str, default=None, help="Keyword for action."
    )
    args = parser.parse_args()

    if args.action == "crawl":
        start_time = time.time()  # 開始時間
        crawl()
        end_time = time.time()
        print(f"It takes {end_time - start_time} sec.")
    elif args.action == "download":
        start_time = time.time()  # 開始時間
        if not os.path.exists("images_0_articles.jsonl"):
            crawl_popular_article()

        folder_name = "images_0"
        if not os.path.exists(folder_name):
            # 如果資料夾不存在，則創建它
            os.makedirs(folder_name)
            print(f"資料夾 '{folder_name}' 已創建成功")
        folder_name = "images_1"
        if not os.path.exists(folder_name):
            # 如果資料夾不存在，則創建它
            os.makedirs(folder_name)
            print(f"資料夾 '{folder_name}' 已創建成功")
        download()
        end_time = time.time()
        print(f"It takes {end_time - start_time} sec.")
    elif args.action == "push":
        if args.start_date is None or args.end_date is None:
            parser.error("The 'push' action requires start_date and end_date.")
        start_time = time.time()  # 開始時間
        push(args.start_date, args.end_date)
        end_time = time.time()
        print(f"It takes {end_time - start_time} sec.")
    elif args.action == "popular":
        if args.start_date is None or args.end_date is None:
            parser.error("The 'popular' action requires start_date and end_date.")
        start_time = time.time()  # 開始時間
        popular(args.start_date, args.end_date)
        end_time = time.time()
        print(f"It takes {end_time - start_time} sec.")

    elif args.action == "keyword":
        if (
            args.start_date is None
            or args.end_date is None
            or args.search_keyword is None
        ):
            parser.error(
                "The 'keyword' action requires start_date, end_date, and search_keyword."
            )
        start_time = time.time()  # 開始時間
        keyword(args.start_date, args.end_date, args.search_keyword)
        end_time = time.time()
        print(f"It takes {end_time - start_time} sec.")
