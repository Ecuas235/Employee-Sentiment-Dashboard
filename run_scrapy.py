from web_crawler.web_crawler.spiders.ambitionCrawler import AmbitioncrawlerSpider
from scrapy.crawler import CrawlerProcess
import pandas as pd

import sys

def ambition_crawler(company_name):
    process = CrawlerProcess(settings = {
        'FEED_URI' : f'../Sentiment Analysis/data/employee_{company_name}.csv',
        'FEED_FORMAT' : 'csv'
    })
    process.crawl(AmbitioncrawlerSpider, company_name=company_name)
    process.start()  
    
if __name__ == "__main__":
    ambition_crawler(sys.argv[1])