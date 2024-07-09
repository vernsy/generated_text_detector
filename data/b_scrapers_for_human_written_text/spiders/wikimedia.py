import scrapy
from scrapy.selector import Selector
import os

# dump of wikimedia from the internet archive from 2015: 
# https://archive.org/details/dewikisource-20151201<<

# this is needed in order to find module setup
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from setup import config

PROJECT_PATH = config.PROJECT_PATH
## input 
INPUT_OS_PATH_HUMAN_WRITTEN_RAW = PROJECT_PATH + config.OS_PATH_HUMAN_WRITTEN_RAW
## output
OUTPUT_FILENAME_HUMAN_WRITTEN_SCRAPED_WIKIPEDIA = config.FILENAME_HUMAN_WRITTEN_SCRAPED_WIKIMEDIA
OUTPUT_OS_PATH_HUMAN_WRITTEN_SCRAPED = PROJECT_PATH + config.OS_PATH_HUMAN_WRITTEN_SCRAPED

class WikimediaSpider(scrapy.Spider):
    name = 'wikimedia' #'german_wikimedia_from_xml_dump'
    
    start_urls = [
        'file://'+ INPUT_OS_PATH_HUMAN_WRITTEN_RAW + '/dewikisource-20151201-pages-articles-multistream-part-1.xml',
        'file://'+ INPUT_OS_PATH_HUMAN_WRITTEN_RAW + '/dewikisource-20151201-pages-articles-multistream-part-2.xml',
        'file://'+ INPUT_OS_PATH_HUMAN_WRITTEN_RAW + '/dewikisource-20151201-pages-articles-multistream-part-3.xml',
        'file://'+ INPUT_OS_PATH_HUMAN_WRITTEN_RAW + '/dewikisource-20151201-pages-articles-multistream-part-4.xml',
        'file://'+ INPUT_OS_PATH_HUMAN_WRITTEN_RAW + '/dewikisource-20151201-pages-articles-multistream-part-5.xml',
        ]

    def parse(self, response):
        print(self)
        xml_content = response.body
        xml_selector = Selector(text=xml_content)

        # XPath selectors
        timestamps = xml_selector.xpath('//timestamp/text()').getall()
        data = xml_selector.xpath('//text/text()').getall()

        texts_to_filter = zip(timestamps,data)

        for timestamp, text in texts_to_filter:
            yield {
                'timestamp': timestamp,
                'text': text,             
            }