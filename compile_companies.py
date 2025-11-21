import requests
from bs4 import BeautifulSoup
import time
import random
import pandas as pd
from tqdm import tqdm

urls = {
    "Groothandel": "https://companyinfo.nl/branche/groothandel-46?page=",
    "Opslag": "https://companyinfo.nl/branche/opslag-521?page="
}

num_pages = None

while num_pages is None:
    try: 
        num_pages = int(input("How many pages would you like to scrape?: "))
    except: 
        print(f"Enter a valid integer: ")
 
company_data = {}
company_counter = 0

for sector, base_url in tqdm(urls, desc="Sectors"):
    print(f"\nScraping sector {sector}")
    for page in tqdm.tqdm(range(0, num_pages)):
        url_page = base_url + str(page)
        print(f"Fetching {url_page} ...")
        time.sleep(random.uniform(2, 4))  # try not to get barred 
        
        resp = requests.get(url_page)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        companies = []

        # Find company card wrappers
        for a in soup.find_all("a", class_="hover:cursor-pointer hover:no-underline"):
            href = a.get("href", "")
            if not href.startswith("/organisatieprofiel/"):
                continue
            
            # Extract the company name
            name_div = a.find("div", class_="title-6")
            name = name_div.get_text(strip=True) if name_div else None
            
            # Extract the location
            loc_div = a.find("div", class_="text-page-foreground-light/50 text-sm")
            location = loc_div.get_text(strip=True) if loc_div else None

            if name:
                company_info = {
                    "id": company_counter,
                    "name": name,
                    "location": location,
                }
                companies.append(company_info)
                company_data[company_counter] = company_info
                company_counter += 1
        if not companies:
            print(f"No companies found on page {page}, stopping.")
            break

# convert to df
df = pd.DataFrame(company_data.values())

# save to csv
df.to_csv("companies.csv", index=False, encoding="utf-8")

print(f"Saved {len(df)} companies to companies.csv")

