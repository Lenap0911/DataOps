import os
import csv
import json
import hashlib
import time
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import boto3

# Initialize AWS Bedrock client
client = boto3.client('bedrock-runtime', region_name='your-region')  # replace with your AWS region

# --- Helper functions ---

def company_id(name, location):
    """Generate unique ID from company name + location"""
    key = f"{name}|{location}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()

def fetch_page_text(url):
    """Fetch and extract text from webpage"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        r = requests.get(url, timeout=15, headers=headers)
        r.raise_for_status()
        html = r.text
        soup = BeautifulSoup(html, "html.parser")
        for element in soup(["script", "style", "noscript", "iframe", "nav", "footer"]):
            element.decompose()
        text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text, html
    except Exception as e:
        print(f"Fetch error for {url}: {e}")
        return "", ""

def search_company_website(company_name, location):
    """Stub for website search - returns 'unknown' for simplicity"""
    # For actual use, integrate Google API or other search method
    return "unknown"

def call_bedrock_extract(text, company_name):
    """Call AWS Bedrock to extract structured data from website content"""
    prompt_template = f"""
    Extract the following information from the company's website content:

    - Pallets direct (number)
    - Warehouse size in m2
    - Temperature (cold, ambient, unknown)
    - Family owned (yes/no)
    - WMS terms (list)

    Content:
    {text}

    Provide the output in JSON format with keys:
    pallets_direct, warehouse_m2, temperature, family_owned, wms_terms
    """

    try:
        response = client.invoke_model(
            modelId='your-bedrock-model-id',  # Replace with your Bedrock model ID
            body=json.dumps({"prompt": prompt_template, "maxTokens": 500}),
            contentType='application/json'
        )
        result_bytes = response['body'].read()
        result_str = result_bytes.decode('utf-8')
        result_json = json.loads(result_str)
        return {
            "pallets_direct": result_json.get('pallets_direct'),
            "warehouse_m2": result_json.get('warehouse_m2'),
            "temperature": result_json.get('temperature', 'unknown'),
            "family_owned": result_json.get('family_owned', None),
            "wms_terms": result_json.get('wms_terms', []),
            "evidence": "Extracted via AWS Bedrock"
        }
    except Exception as e:
        print(f"Error calling Bedrock: {e}")
        return {
            "pallets_direct": None,
            "warehouse_m2": None,
            "temperature": "unknown",
            "family_owned": None,
            "wms_terms": [],
            "evidence": f"Error: {e}"
        }

def compute_score(extract):
    """Compute relevancy score based on extracted data"""
    score = 0
    pallets = extract.get('pallets_direct') or 0
    m2 = extract.get('warehouse_m2') or 0

    if pallets == 0 and m2 > 0:
        pallets = int(m2 / 1.35)

    # Size score
    if pallets >= 10000:
        score += 35
    elif pallets >= 7000:
        score += 28
    elif pallets >= 4000:
        score += 18
    elif pallets >= 1000:
        score += 10
    elif pallets > 0:
        score += 5

    # Business type score
    wms = extract.get('wms_terms') or []
    if len(wms) >= 3:
        score += 25
    elif len(wms) >= 1:
        score += 20
    else:
        score += 10

    # Family owned
    if extract.get('family_owned') is True:
        score += 10
    elif extract.get('family_owned') is False:
        score += 5
    else:
        score += 7  # neutral

    # Temperature
    temp = extract.get('temperature', 'unknown')
    if temp == 'cold':
        score += 10
    elif temp == 'ambient':
        score += 6
    else:
        score += 4

    # WMS signals
    pain_score = min(len(wms) * 5, 20)
    score += pain_score

    return min(score, 100)

# --- Main processing ---

def main():
    input_csv = 'companies.csv'
    output_csv = 'scored_companies.csv'
    results = []

    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get('name') or row.get('company') or row.get('Company')
            location = row.get('location') or row.get('Location') or ""

            if not name:
                print(f"Skipping row with no name: {row}")
                continue

            # Get website
            website = row.get('website') or row.get('Website')
            if not website:
                print(f"No website info for {name}, searching...")
                website = search_company_website(name, location)
                if not website:
                    website = "unknown"

            # Fetch website content if URL is valid
            text = ""
            if website and website != "unknown":
                print(f"Fetching: {website}")
                text, _ = fetch_page_text(website)
            else:
                print(f"No valid website for {name}, marking as unknown")
                text = ""

            # Extract info
            extract = {}
            if text:
                extract = call_bedrock_extract(text, name)
            else:
                extract = {
                    "pallets_direct": None,
                    "warehouse_m2": None,
                    "temperature": "unknown",
                    "family_owned": None,
                    "wms_terms": [],
                    "evidence": "No website content"
                }

            # Estimate pallets if missing
            if extract.get('pallets_direct') is None and extract.get('warehouse_m2'):
                extract['pallets_direct'] = int(extract['warehouse_m2'] / 1.35)
                extract['pallet_source'] = "estimated_from_m2"
            else:
                extract['pallet_source'] = "direct" if extract.get('pallets_direct') else "none"

            score = compute_score(extract)

            rec = {
                "company_id": company_id(name, location),
                "name": name,
                "location": location,
                "website": website if website != "unknown" else "unknown",
                "pallet_estimate": extract.get('pallets_direct'),
                "pallet_source": extract.get('pallet_source'),
                "warehouse_m2": extract.get('warehouse_m2'),
                "temperature": extract.get('temperature'),
                "family_owned": extract.get('family_owned'),
                "wms_terms": ";".join(extract.get('wms_terms', [])),
                "evidence": extract.get('evidence', ''),
                "relevance_score": score,
                "last_updated": datetime.utcnow().isoformat() + "Z"
            }

            results.append(rec)

    # Sort results by relevance
    results_sorted = sorted(results, key=lambda x: x['relevance_score'], reverse=True)

    # Save to output CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = list(results_sorted[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_sorted)

    print(f"Scored companies saved to {output_csv}")

if __name__ == "__main__":
    main()