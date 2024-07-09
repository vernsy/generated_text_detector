import scrapy
import os
import re
import locale
from typing import List
from datetime import datetime

# this is needed in order to find module setup
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from setup import config

PROJECT_PATH = config.PROJECT_PATH
## input 
INPUT_OS_PATH_HUMAN_WRITTEN_RAW = '/data/verena/raw_datasets/wikipedia-de-html/de/articles' # on nofake
## output
OUTPUT_FILENAME_HUMAN_WRITTEN_SCRAPED_WIKIPEDIA = config.FILENAME_HUMAN_WRITTEN_SCRAPED_WIKIPEDIA
OUTPUT_OS_PATH_HUMAN_WRITTEN_SCRAPED = PROJECT_PATH + config.OS_PATH_HUMAN_WRITTEN_SCRAPED

# info: the dump is manually downloaded from here:
# https://dumps.wikimedia.org/other/static_html_dumps/current/de/
# we chose from 2008 because then it was an easier filestructure to scrape

class WikipediaSpider(scrapy.Spider):
    name = 'wikipedia'
    
    def start_requests(self):
        all_html_files = get_html_files(INPUT_OS_PATH_HUMAN_WRITTEN_RAW)
        for file in all_html_files:
            yield scrapy.Request('file://' + file, callback=self.parse_article)

    def parse_article(self, response) -> None:
        date = False
        extracted_content = False

        ## Extract the content out of wikipedia template
        content = response.xpath(
            '//comment()[contains(., "start content")]/following-sibling::node()[following-sibling::div[@class="printfooter"]]//text()'
            ).getall()
        extracted_content = ' '.join(content).strip()


        ### date: 
        li_content = response.css('li#f-credits::text').get()
        if li_content:
            date_match = re.search(r'am (\d+\. [a-zA-Z]+ \d+)', li_content)
            if date_match:
                extracted_date = date_match.group(1)
                date = convert_creation_date(extracted_date)
        
        if date and extracted_content and len(extracted_content) > 50:
            yield {
                'timestamp': date,
                'text': extracted_content,
            }
    
def get_html_files(directory) -> List[str]:
    html_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".html"):
                html_files.append(os.path.join(root, file))
    return html_files

def convert_creation_date(timestamp_string: str) -> str:
    locale.setlocale(locale.LC_TIME, 'de_DE.utf-8')
    print(timestamp_string)
    timestamp_string = timestamp_string.strip()
    date_format = "%d. %B %Y"
    try:
        timestamp = datetime.strptime(timestamp_string, date_format)
    except:
        print("ERROR for converting ",timestamp_string)
        return False

    return timestamp.strftime("%Y-%m-%d")