import scrapy


class AmbitioncrawlerSpider(scrapy.Spider):
    name = "ambitionCrawler"
    #start_urls = [f'https://www.ambitionbox.com/reviews/{}-reviews']
    page = 2
    def __init__(self, company_name,*args, **kwargs):
        super(AmbitioncrawlerSpider, self).__init__(*args, **kwargs)
        self.start_urls = [f'https://www.ambitionbox.com/reviews/{company_name}-reviews']
        self.company_name = company_name
        
    def parse(self, response):
            reviews = response.css("div.ab_comp_review_card")
            for review in reviews:
                title = review.css("h2.bold-title-s.review-title::text").extract()[0]
                ratings = review.css("span.avg-rating.bold-Labels::text").extract()
                date = review.css("span.status.caption-subdued::text").extract()
                likes = review.css(".overflow-wrap:nth-child(2)::text").extract()
                dislikes = review.css(".overflow-wrap:nth-child(4)::text").extract()
                yield {
                    'title':title,
                    'Rating': ratings,
                    'date' : date,
                    'likes' : likes,
                    'dislikes' : dislikes
                    }
            next_page = f"https://www.ambitionbox.com/reviews/{self.company_name}-reviews?page="+ str(AmbitioncrawlerSpider.page)
            if next_page is not f"https://www.ambitionbox.com/reviews/{self.company_name}-reviews":
                AmbitioncrawlerSpider.page += 1
                yield response.follow(next_page, self.parse)