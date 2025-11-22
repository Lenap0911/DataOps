import os
import csv
import json
import hashlib
import re
import urllib.request
from datetime import datetime
from typing import Dict, Any, Tuple, List

import boto3

# ---------- AWS Clients & Local Config ----------
# Note: You MUST configure AWS credentials locally for boto3 to work.
bedrock = boto3.client("bedrock-runtime")

# LOCAL CONFIGURATION (Replaces Environment Variables)
INPUT_FILE = "companies.csv" # Local file path
BEDROCK_MODEL = "anthropic.claude-3-5-sonnet-20241022-v2:0"
MAX_COMPANIES = 20  # limit for demo


# ========== GABY'S SCORING LOGIC (DEDUCTION MODEL) ==========
# (Functions 'score_lead_deduction' and 'make_sales_note' remain unchanged)

def score_lead_deduction(features: Dict[str, Any]) -> Tuple[int, str, bool]:
    """Calculate lead score (0-100) and segment (A/B/C) using deduction method."""
    if not features.get("has_warehouse", False):
        return 0, "", False
    score = 100
    # 1. Warehouse type
    wtype = (features.get("warehouse_type") or "unknown").lower()
    if wtype == "freezer": pass
    elif wtype == "mixed": score -= 10
    elif wtype == "ambient": score -= 20
    else: score -= 30
    # 2. Scale
    scale = (features.get("approx_scale") or "unknown").lower()
    if scale == "large": pass
    elif scale == "medium": score -= 10
    elif scale == "small": score -= 25
    # 3. Pallet capacity
    pallets = features.get("approx_pallet_capacity")
    try: pallets = int(pallets) if pallets is not None else 0
    except (ValueError, TypeError): pallets = 0
    if pallets > 30000: pass
    elif 7000 <= pallets <= 30000: score -= 5
    elif 1 <= pallets < 7000: score -= 15
    else: score -= 25
    # 4. Industry
    industry = (features.get("industry") or "").lower()
    ideal_keywords = ["food", "fish", "frozen", "pharma", "logistics", "3pl", "cold chain"]
    ok_keywords = ["manufactur", "bouw", "construction", "wholesale", "groothandel"]
    if any(k in industry for k in ideal_keywords): pass
    elif any(k in industry for k in ok_keywords): score -= 10
    elif industry: score -= 40
    # 5. Public sector
    if features.get("is_public_sector"): score -= 5
    # 6. Safety/compliance focus
    if not features.get("safety_focus", False): score -= 10
    # 7. Confidence in extraction
    conf = features.get("website_confidence")
    try: conf = float(conf) if conf is not None else 0.5
    except (ValueError, TypeError): conf = 0.5
    if conf < 0.5: score -= 25
    
    score = max(0, min(100, score)) # Clamp
    
    if score >= 80: segment = "A"
    elif score >= 60: segment = "B"
    elif score >= 40: segment = "C"
    else: segment = ""
    is_interesting = score >= 40
    return score, segment, is_interesting


def make_sales_note(features: Dict[str, Any]) -> str:
    """Create human-friendly note for sales team."""
    wtype = features.get("warehouse_type", "unknown")
    pallets = features.get("approx_pallet_capacity", "?")
    industry = features.get("industry", "unknown")
    scale = features.get("approx_scale", "unknown")
    freezer_flag = "freezer" in (wtype or "").lower()
    parts = [
        f"{wtype or 'unknown'} warehouse",
        f"({scale})" if scale else None,
        f"~{pallets} pallets",
        f"industry: {industry}",
        "[HIGH PRIORITY: freezer]" if freezer_flag else None
    ]
    return ", ".join(str(p) for p in parts if p)


# ========== HELPER FUNCTIONS ==========
def company_id(name: str, location: str) -> str:
    """Generate unique ID from company name + location."""
    key = f"{name}|{location}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()

def fetch_website_html(url: str) -> str:
    """Fetch website HTML as text (basic)."""
    if not url: return ""
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            raw_html = resp.read().decode("utf-8", errors="ignore")
        return raw_html
    except Exception as e:
        print(f"✗ Fetch error for {url}: {e}")
        return ""

def strip_html_tags(html: str) -> str:
    """Very simple HTML → text conversion, no external libs."""
    if not html: return ""
    html = re.sub(r"<script.*?>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style.*?>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ========== BEDROCK LLM EXTRACTION (Functions remain largely the same, but using global model ID) ==========
def build_extraction_prompt(company_name: str, location: str, website_text: str) -> str:
    """Build structured extraction prompt for Bedrock."""
    # (Same prompt logic as original)
    text_sample = website_text[:12000]  # limit tokens

    return f"""You are helping a sales team that sells autonomous drone-based stock counting to warehouses and logistics companies.

Your job: Extract structured features from this company's website to determine if they are a good sales lead.

IMPORTANT:
- Only use information clearly stated in the text.
- If uncertain, set values to "unknown"/null/false and lower confidence instead of guessing.

Company:
- Name: {company_name}
- Location: {location}

Website Content:
\"\"\"{text_sample}\"\"\"

Extract the following as a JSON object:

{{
  "has_warehouse": <bool - true if they clearly operate physical warehouses/DCs/storage facilities>,
  "warehouse_type": <"freezer" | "mixed" | "ambient" | "unknown">,
  "approx_scale": <"large" | "medium" | "small" | "unknown">,
  "approx_pallet_capacity": <int or null - best estimate of total pallet positions>,
  "industry": <short string e.g. "frozen food logistics", "3PL", "pharma distribution">,
  "is_public_sector": <bool - true for government/municipality/public entities>,
  "safety_focus": <bool - true if emphasis on safety/compliance/audits/certifications like HACCP, ISO, BRC>,
  "website_confidence": <float 0.0-1.0 - how confident you are in these findings. 1.0=very clear, 0.3=vague, 0.1=almost no info>
}}

Return ONLY the JSON object, no explanations.
"""

def heuristic_extract(text: str, company_name: str) -> Dict[str, Any]:
    """Fallback: simple regex-based extraction if LLM fails."""
    # (Same heuristic logic as original)
    print(f"⚠ Using heuristic extraction for {company_name}")
    result = {
        "has_warehouse": False, "warehouse_type": "unknown", "approx_scale": "unknown",
        "approx_pallet_capacity": None, "industry": "", "is_public_sector": False,
        "safety_focus": False, "website_confidence": 0.3,
    }
    text_lower = text.lower()
    warehouse_terms = ["warehouse", "distribution center", "dc", "storage facility", "logistics center"]
    if any(term in text_lower for term in warehouse_terms): result["has_warehouse"] = True
    if re.search(r"freez|cold\s+chain|refrigerat|cold\s+storage|temperature\s+controlled", text_lower): result["warehouse_type"] = "freezer"
    elif re.search(r"ambient|dry\s+storage", text_lower): result["warehouse_type"] = "ambient"
    if re.search(r"3pl|third\s+party\s+logistics", text_lower): result["industry"] = "3PL logistics"
    elif re.search(r"food|frozen|fresh", text_lower): result["industry"] = "food logistics"
    elif re.search(r"pharma|medical", text_lower): result["industry"] = "pharma distribution"
    if re.search(r"haccp|iso|brc|ifs|safety|compliance|certified", text_lower): result["safety_focus"] = True
    if re.search(r"government|municipality|gemeente|public\s+sector", text_lower): result["is_public_sector"] = True
    return result


def call_bedrock_extract(company_name: str, location: str, website_text: str) -> Dict[str, Any]:
    """Call Bedrock (Claude) to extract structured warehouse features."""
    if not website_text or len(website_text) < 50:
        print(f"⚠ Insufficient website content for {company_name}")
        return heuristic_extract(website_text, company_name)
    prompt = build_extraction_prompt(company_name, location, website_text)
    try:
        body = {
            "anthropic_version": "bedrock-2023-05-31", "max_tokens": 800, "temperature": 0.1,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        }
        response = bedrock.invoke_model(
            modelId=BEDROCK_MODEL,
            body=json.dumps(body),
        )
        result = json.loads(response["body"].read().decode("utf-8"))
        text_response = result["content"][0]["text"].strip()
        text_response = re.sub(r"```json\s*|\s*```", "", text_response).strip()
        features = json.loads(text_response)
        print(f"✓ Bedrock extraction successful for {company_name}")
        return features
    except Exception as e:
        print(f"✗ Bedrock error for {company_name}: {e}")
        return heuristic_extract(website_text, company_name)


# ========== MAIN LOCAL ENTRY POINT (Replaces lambda_handler) ==========

def run_lead_scorer_local():
    """
    Local script to read companies.csv, process leads, and print the results.
    """
    print("=" * 60)
    print(f"Local Lead Scoring Tool started at {datetime.utcnow().isoformat()}Z")
    print("=" * 60)
    print(f"📁 Reading leads from local file: {INPUT_FILE}")

    try:
        with open(INPUT_FILE, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            results: List[Dict[str, Any]] = []
            processed = 0
            errors = 0

            for row in reader:
                if processed >= MAX_COMPANIES:
                    break

                try:
                    name = row.get("name") or row.get("company") or row.get("Company")
                    location = row.get("location") or row.get("Location") or row.get("city", "")
                    website = row.get("website") or row.get("Website")

                    if not name:
                        print(f"⚠ Skipping row with no name: {row}")
                        continue

                    cid = company_id(name, location)
                    print("\n" + "─" * 60)
                    print(f"🏢 Processing: {name} ({location})")
                    print(f"   ID: {cid}")
                    print(f"   🌐 Website: {website or 'N/A'}")

                    html = fetch_website_html(website) if website else ""
                    text = strip_html_tags(html)

                    # 1. Feature Extraction (LLM or Heuristic)
                    features = call_bedrock_extract(name, location, text)
                    
                    # 2. Scoring and Segmentation
                    score, segment, is_interesting = score_lead_deduction(features)
                    sales_note = make_sales_note(features)

                    emoji = (
                        "🔥" if segment == "A" else
                        "✓" if segment == "B" else
                        "○" if segment == "C" else
                        "✗"
                    )
                    print(f"   {emoji} Score: {score}/100 | Segment: {segment or 'Not interesting'}")
                    print(f"   📝 {sales_note}")

                    rec = {
                        "company_id": cid,
                        "name": name,
                        "location": location,
                        "website": website,
                        "score": score,
                        "segment": segment,
                        "is_interesting": is_interesting,
                        "sales_note": sales_note,
                        "features": features, # Include extracted features for analysis
                    }

                    results.append(rec)
                    processed += 1

                except Exception as e:
                    errors += 1
                    print(f"✗ Error processing company {name}: {e}")
                    continue

            print("\n" + "=" * 60)
            print("📊 SUMMARY")
            print(f"   Total processed: {processed}")
            print(f"   Errors: {errors}")
            print("=" * 60)
            
            return results

    except FileNotFoundError:
        print(f"💥 FATAL ERROR: Input file '{INPUT_FILE}' not found. Please create {INPUT_FILE} with company data.")
        return []
    except Exception as e:
        print(f"💥 FATAL ERROR: {e}")
        return []


if __name__ == "__main__":
    final_scores = run_lead_scorer_local()
    
    # Save the final dataset to a JSON file
    output_filename = "scored_companies_dataset.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(final_scores, f, indent=2)
    
    print(f"\n🎉 Processing complete. Full results saved to {output_filename}")
    