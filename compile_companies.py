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
company_counter = 1

for sector, base_url in tqdm(urls.items(), desc="Sectors"):
    print(f"\nScraping sector {sector}")
    for page in tqdm(range(1, num_pages)):
        url_page = base_url + str(page)
        print(f"Fetching {url_page} ...")
        time.sleep(random.uniform(2, 4))  # try not to get barred 
        
        resp = requests.get(url_page)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        companies = []

        # Find company card wrappers
        for a in tqdm(soup.find_all("a", class_="hover:cursor-pointer hover:no-underline"), desc=f"Processing companies on page {page}"):
            href = a.get("href", "")
            if not href.startswith("/organisatieprofiel/"):
                continue
            
            # Extract the company name
            name_div = a.find("div", class_="title-6")
            name = name_div.get_text(strip=True) if name_div else None
            
            # Extract the location
            loc_div = a.find("div", class_="text-page-foreground-light/50 text-sm")
            location = loc_div.get_text(strip=True) if loc_div else None

            # Navigate to company data page
            company_data_url = a.get("href", "")
            comp_resp = requests.get("https://companyinfo.nl" + company_data_url)
            comp_resp.raise_for_status()
            comp_soup = BeautifulSoup(comp_resp.text, "html.parser")

            website = None

            # Find div with contact information
            contact_div = None
            for div in comp_soup.find_all("div", class_=lambda x: x and "flex" in x or True):
                if div.get_text(strip=True).startswith("Contact gegevens"):
                    contact_div = div
                    break

            if contact_div:
                # Find company website
                a_tag = contact_div.find("a", href=True)
                if a_tag:
                    website = a_tag["href"]
                    if website.startswith("//"):
                        website = "https:" + website
            # store company data in dataframe
            if name:
                company_info = {
                    "id": company_counter,
                    "name": name,
                    "location": location,
                    "website": website
                }
                companies.append(company_info)
                company_data[company_counter] = company_info
                company_counter += 1
        if not companies:
            print(f"No companies found on page {page}, stopping.")
            break

# store in json format
df = pd.DataFrame(company_data.values())
df.to_csv("companies.csv", index=False, encoding="utf-8")

print(f"Saved {len(df)} companies to companies.csv")