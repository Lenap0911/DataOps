import os
import csv
import json
import hashlib
import time
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime

# AWS clients - commented out since we're processing locally
# import boto3
# s3 = boto3.client("s3")
# dynamodb = boto3.resource("dynamodb")
# bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")  # Change region if needed

# Environment variables or defaults
LEADS_TABLE = os.environ.get("LEADS_TABLE", "Leads")
# BUCKET = os.environ.get("BUCKET", "my-leads-bucket")
# BEDROCK_MODEL = os.environ.get("BEDROCK_MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0")

# For local processing, comment out AWS clients
# table = dynamodb.Table(LEADS_TABLE)

# ---------- Helper Functions ----------
def company_id(name, location):
    """Generate unique ID from company name + location"""
    key = f"{name}|{location}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()

def search_company_website(company_name, location):
    """
    Search for company website using Google Custom Search API
    Set GOOGLE_API_KEY and GOOGLE_CX in environment variables
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    cx = os.environ.get("GOOGLE_CX")
    
    if not api_key or not cx:
        print(f"Warning: Google Search API not configured for {company_name}")
        return None
    
    try:
        q = f"{company_name} {location} warehouse"
        url = (f"https://www.googleapis.com/customsearch/v1"
               f"?key={api_key}&cx={cx}&q={requests.utils.quote(q)}")
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        items = r.json().get("items", [])
        for item in items:
            link = item.get('link')
            if link and link.startswith('http'):
                print(f"Found website for {company_name}: {link}")
                return link
    except Exception as e:
        print(f"Search error for {company_name}: {e}")
    return None

def fetch_page_text(url):
    """Fetch and extract text from webpage"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        r = requests.get(url, timeout=15, headers=headers)
        r.raise_for_status()
        html = r.text

        # Parse with BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for element in soup(["script", "style", "noscript", "iframe", "nav", "footer"]):
            element.decompose()
        text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text, html
    except Exception as e:
        print(f"Fetch error for {url}: {e}")
        return "", ""

# ---------- LLM Extraction with Bedrock ----------
def call_bedrock_extract(text, company_name):
    """
    Call Amazon Bedrock (Claude) to extract structured data from website text
    """
    text_sample = text[:20000]
    prompt = f"""You are a data extraction specialist. Analyze the following website text from "{company_name}" and extract warehouse/logistics information.

Website Text:
{text_sample}

Extract the following information and respond with ONLY a valid JSON object (no markdown, no explanation):

{{
  "pallets_direct": <integer if explicit pallet count/positions mentioned, else null>,
  "warehouse_m2": <integer if square meters (m2/m²/sqm) mentioned, else null>,
  "temperature": "<'cold' if refrigerated/freezer/cold-chain mentioned, 'ambient' if normal temp, else 'unknown'>",
  "family_owned": <true if 'family-owned' or 'family business' explicitly stated, false if corporate/public company indicated, else null>,
  "wms_terms": [<list of warehouse automation terms found: 'WMS', 'AS/RS', 'automated storage', 'shuttle system', 'VNA', 'AGV', 'robotics', etc.>],
  "evidence": "<brief quotes or phrases supporting your findings, max 200 chars>"
}}

Be conservative: only extract what is explicitly stated. If uncertain, use null."""

    try:
        # Call Bedrock (simulate with placeholder if needed)
        # For local testing, comment out actual call and simulate
        # response = bedrock.invoke_model(...)
        # Simulate response:
        # response_body = json.loads(response['body'].read())

        # Placeholder: simulate no extraction
        # Remove or replace with actual AWS call if available
        # Here, for illustration, we return empty extraction:
        extracted = {
            "pallets_direct": None,
            "warehouse_m2": None,
            "temperature": "unknown",
            "family_owned": None,
            "wms_terms": [],
            "evidence": ""
        }
        return extracted
    except Exception as e:
        print(f"Bedrock extraction error for {company_name}: {e}")
        return heuristic_extract(text, company_name)

def heuristic_extract(text, company_name):
    """Fallback heuristic extraction"""
    print(f"Using heuristic extraction for {company_name}")
    result = {
        "pallets_direct": None,
        "warehouse_m2": None,
        "temperature": "unknown",
        "family_owned": None,
        "wms_terms": [],
        "evidence": ""
    }
    text_lower = text.lower()
    # Extract warehouse area
    m2_match = re.search(r'(\d{1,3}(?:[,\s]\d{3})*|\d+)\s*(?:m2|m²|sqm|square\s+meters)', text_lower)
    if m2_match:
        m2_str = m2_match.group(1).replace(',', '').replace(' ', '')
        try:
            result['warehouse_m2'] = int(m2_str)
            result['evidence'] += f"Area: {m2_match.group(0)}; "
        except:
            pass
    # Pallet count
    pallet_match = re.search(r'(\d{1,3}(?:[,\s]\d{3})*|\d+)\s*(?:pallets?|pallet\s+positions?|pallet\s+places?)', text_lower)
    if pallet_match:
        pallet_str = pallet_match.group(1).replace(',', '').replace(' ', '')
        try:
            result['pallets_direct'] = int(pallet_str)
            result['evidence'] += f"Pallets: {pallet_match.group(0)}; "
        except:
            pass
    # Temperature
    if re.search(r'cold\s+chain|refrigerat|freezer|cold\s+storage|temperature\s+controlled', text_lower):
        result['temperature'] = "cold"
        result['evidence'] += "Cold storage detected; "
    # Family owned
    if re.search(r'family[\s-]owned|family\s+business|family\s+run', text_lower):
        result['family_owned'] = True
        result['evidence'] += "Family-owned; "
    # WMS terms
    automation_terms = ['wms', 'warehouse management system', 'as/rs', 'asrs', 
                       'automated storage', 'shuttle', 'vna', 'very narrow aisle',
                       'agv', 'automated guided vehicle', 'robot', 'automation']
    for term in automation_terms:
        if term in text_lower:
            if term.upper() not in result['wms_terms']:
                result['wms_terms'].append(term.upper())
    # Limit list length
    result['wms_terms'] = list(set(result['wms_terms']))[:5]
    return result

# ---------- Scoring Function ----------
def compute_score(extract):
    score = 0
    pallets = extract.get('pallets_direct') or 0
    m2 = extract.get('warehouse_m2') or 0
    # Estimate pallets if not available
    if not pallets and m2:
        try:
            pallets = int(m2 / 1.35)
        except:
            pallets = 0
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
    # Business type
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
        score += 7
    # Temperature
    temp = extract.get('temperature', 'unknown')
    if temp == 'cold':
        score += 10
    elif temp == 'ambient':
        score += 6
    else:
        score += 4
    # WMS signals
    pain_score = min(len(wms)*5, 20)
    score += pain_score
    return min(score, 100)

# ---------- Main Processing ----------
def process_leads(input_csv='input/leads.csv', output_csv='scored_leads.csv'):
    df = pd.read_csv(input_csv)
    results = []

    for index, row in df.iterrows():
        # Read fields
        name = row['name']
        location = row['loc']
        website = row.get('website', None)

        cid = company_id(name, location)

        # If website not provided, search
        if not pd.notna(website) or not website:
            print(f"Searching website for {name}...")
            website = search_company_website(name, location)

        # Fetch website content
        text = ""
        if website:
            print(f"Fetching webpage for {name}: {website}")
            text, html = fetch_page_text(website)
            # Save raw HTML if needed (skipped here)

        # Extract data
        extract = {}
        if text:
            extract = call_bedrock_extract(text, name)
        else:
            # Mark variables as unknown if no content
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
            try:
                extract['pallets_direct'] = int(extract['warehouse_m2'] / 1.35)
            except:
                extract['pallets_direct'] = None
            extract['pallet_source'] = "estimated_from_m2"
        else:
            extract['pallet_source'] = "direct" if extract.get('pallets_direct') else "none"

        # Compute relevance score
        score = compute_score(extract)

        # Prepare output record
        record = {
            'id': row['id'],
            'name': name,
            'location': location,
            'website': website or 'unknown',
            'pallets': extract.get('pallets_direct'),
            'warehouse_m2': extract.get('warehouse_m2'),
            'temperature': extract.get('temperature'),
            'family_owned': extract.get('family_owned'),
            'wms_terms': ';'.join(extract.get('wms_terms', [])),
            'evidence': extract.get('evidence'),
            'relevance_score': score
        }
        results.append(record)
        print(f"{name}: Score={score}")

    # Save to CSV
    df_out = pd.DataFrame(results)
    df_out.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

# Run the process
if __name__ == "__main__":
    process_leads(input_csv='input/leads.csv', output_csv='scored_leads.csv')