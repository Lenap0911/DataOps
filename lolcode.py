import os
import csv
import json
import hashlib
import re
import urllib.request
from datetime import datetime
from typing import Dict, Any, Tuple, List

import boto3

# ---------- AWS Clients & Environment Variables (Kept as is) ----------
s3 = boto3.client("s3")
bedrock = boto3.client("bedrock-runtime")

BUCKET = os.environ.get("BUCKET", "website-scoring-bucket")
INPUT_KEY = os.environ.get("INPUT_KEY", "companies.csv")
BEDROCK_MODEL = os.environ.get(
    "BEDROCK_MODEL",
    "anthropic.claude-3-5-sonnet-20241022-v2:0"
)
MAX_COMPANIES = int(os.environ.get("MAX_COMPANIES", "20"))  # limit for demo

# ========== GABY'S SCORING LOGIC (DEDUCTION MODEL) - Kept for reference/audit ==========

def score_lead_deduction(features: Dict[str, Any]) -> Tuple[int, str, bool]:
    """
    Calculate lead score (0-100) and segment (A/B/C) using deduction method.
    This function is primarily for validation/audit in the refactored design,
    as the LLM is now instructed to perform the scoring.
    """
    # Must have a warehouse to be a lead
    if not features.get("has_warehouse", False):
        return 0, "", False

    score = 100

    # 1. Warehouse type (freezer = ideal, unknown = worst)
    wtype = (features.get("warehouse_type") or "unknown").lower()
    if wtype == "freezer":
        pass  # Perfect - no penalty
    elif wtype == "mixed":
        score -= 10
    elif wtype == "ambient":
        score -= 20
    else:  # unknown
        score -= 30

    # 2. Scale (small is less interesting)
    scale = (features.get("approx_scale") or "unknown").lower()
    if scale == "large":
        pass
    elif scale == "medium":
        score -= 10
    elif scale == "small":
        score -= 25

    # 3. Pallet capacity (>=7k ideal, >30k best)
    pallets = features.get("approx_pallet_capacity")
    try:
        pallets = int(pallets) if pallets is not None else 0
    except (ValueError, TypeError):
        pallets = 0

    if pallets > 30000:
        pass
    elif 7000 <= pallets <= 30000:
        score -= 5
    elif 1 <= pallets < 7000:
        score -= 15
    else:  # 0 or unknown
        score -= 25

    # 4. Industry (food/pharma/3PL ideal, construction ok, services bad)
    industry = (features.get("industry") or "").lower()
    ideal_keywords = ["food", "fish", "frozen", "pharma", "logistics", "3pl", "cold chain"]
    ok_keywords = ["manufactur", "bouw", "construction", "wholesale", "groothandel"]

    if any(k in industry for k in ideal_keywords):
        pass  # Ideal - no penalty
    elif any(k in industry for k in ok_keywords):
        score -= 10
    elif industry:
        score -= 40  # Service industry - not interesting

    # 5. Public sector (longer sales cycles)
    if features.get("is_public_sector"):
        score -= 5

    # 6. Safety/compliance focus (important for drone services)
    if not features.get("safety_focus", False):
        score -= 10

    # 7. Confidence in extraction (penalize uncertainty)
    conf = features.get("website_confidence")
    try:
        conf = float(conf) if conf is not None else 0.5
    except (ValueError, TypeError):
        conf = 0.5

    if conf < 0.5:
        score -= 25

    # Clamp to [0, 100]
    score = max(0, min(100, score))

    # Segment assignment
    if score >= 80:
        segment = "A"  # Core target - freezer/large/ideal
    elif score >= 60:
        segment = "B"  # Good targets - worth calling
    elif score >= 40:
        segment = "C"  # Secondary - manual counting approach
    else:
        segment = ""  # Not interesting

    is_interesting = score >= 40

    return score, segment, is_interesting


def make_sales_note(features: Dict[str, Any]) -> str:
    """Create human-friendly note for sales team. Kept for reference/audit."""
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


# ========== HELPER FUNCTIONS (Kept as is) ==========

def company_id(name: str, location: str) -> str:
    """Generate unique ID from company name + location."""
    key = f"{name}|{location}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def fetch_website_html(url: str) -> str:
    """Fetch website HTML as text (basic)."""
    if not url:
        return ""
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            raw_html = resp.read().decode("utf-8", errors="ignore")
        return raw_html
    except Exception as e:
        print(f"✗ Fetch error for {url}: {e}")
        return ""


def strip_html_tags(html: str) -> str:
    """Very simple HTML → text conversion, no external libs."""
    if not html:
        return ""
    # Remove script/style
    html = re.sub(r"<script.*?>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style.*?>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove all tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ========== REFACTORED BEDROCK LLM EXTRACTION & SCORING ==========

def build_combined_prompt(company_name: str, location: str, website_text: str) -> str:
    """
    Build structured prompt for Bedrock to perform:
    1. Feature Extraction
    2. Deductive Scoring
    3. Sales Note Generation
    """
    text_sample = website_text[:12000]  # limit tokens

    # --- SCORING LOGIC INSTRUCTIONS FOR LLM ---
    scoring_logic = """
    # SCORING LOGIC (Deduction Model - Start at 100)
    1. Base Requirement: Must operate a physical warehouse (`has_warehouse=True`). If False, final score is 0 and segment is "".
    2. Warehouse Type:
        - "freezer": 0 points deducted.
        - "mixed": 10 points deducted.
        - "ambient": 20 points deducted.
        - "unknown": 30 points deducted.
    3. Scale:
        - "large": 0 points deducted.
        - "medium": 10 points deducted.
        - "small": 25 points deducted.
        - "unknown": 0 points deducted.
    4. Pallet Capacity (approx_pallet_capacity):
        - > 30000: 0 points deducted.
        - 7000 to 30000: 5 points deducted.
        - 1 to 6999: 15 points deducted.
        - 0 or null: 25 points deducted.
    5. Industry: (Keywords for Ideal: food, fish, frozen, pharma, logistics, 3pl, cold chain | OK: manufactur, construction, wholesale)
        - Ideal Keywords found: 0 points deducted.
        - OK Keywords found: 10 points deducted.
        - Industry known but irrelevant (e.g., "services"): 40 points deducted.
    6. Public Sector (`is_public_sector`=True): 5 points deducted.
    7. Safety Focus (`safety_focus`=False): 10 points deducted.
    8. Confidence: If `website_confidence` < 0.5: 25 points deducted.
    9. Final Score: Clamp result to [0, 100].
    10. Segment:
        - Score >= 80: "A"
        - 60 <= Score < 80: "B"
        - 40 <= Score < 60: "C"
        - Score < 40: ""
    """
    # ---------------------------------------------

    return f"""You are a logistics lead scoring specialist. Your goal is to analyze the provided company information and website content to **extract features, calculate a lead score**, and **generate a sales note**.

The product is autonomous drone-based stock counting for warehouses.

Company:
- Name: {company_name}
- Location: {location}

Website Content:
\"\"\"{text_sample}\"\"\"

{scoring_logic}

Perform all extraction and calculations, then output the final result as a single JSON object.

The output JSON MUST contain all the fields listed in the REQUIRED OUTPUT JSON SCHEMA.
- The scoring and segment fields MUST be calculated strictly following the Scoring Logic.
- The `sales_note` MUST be a human-friendly summary using the extracted features (warehouse type, scale, pallets, industry, plus a HIGH PRIORITY flag if freezer).

REQUIRED OUTPUT JSON SCHEMA:
{{
  "has_warehouse": <bool>,
  "warehouse_type": <"freezer" | "mixed" | "ambient" | "unknown">,
  "approx_scale": <"large" | "medium" | "small" | "unknown">,
  "approx_pallet_capacity": <int or null>,
  "industry": <short string>,
  "is_public_sector": <bool>,
  "safety_focus": <bool>,
  "website_confidence": <float 0.0-1.0>,
  "score": <int 0-100 - calculated based on the Scoring Logic>,
  "segment": <"A" | "B" | "C" | "" - calculated based on the Scoring Logic>,
  "sales_note": <string - formatted summary for sales>
}}

Return ONLY the JSON object, no explanations or other text.
"""


def call_bedrock_extract_and_score(company_name: str, location: str, website_text: str) -> Dict[str, Any]:
    """Call Bedrock (Claude) to perform feature extraction, scoring, and note generation."""
    if not website_text or len(website_text) < 50:
        print(f"⚠ Insufficient website content for {company_name}")
        # Fallback to a simple, low-confidence heuristic result
        return {
            "has_warehouse": False, "warehouse_type": "unknown", "approx_scale": "unknown",
            "approx_pallet_capacity": None, "industry": "", "is_public_sector": False,
            "safety_focus": False, "website_confidence": 0.1,
            "score": 0, "segment": "", "sales_note": "Insufficient website data."
        }

    prompt = build_combined_prompt(company_name, location, website_text)

    try:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1200,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ],
        }

        response = bedrock.invoke_model(
            modelId=BEDROCK_MODEL,
            body=json.dumps(body),
        )

        result = json.loads(response["body"].read().decode("utf-8"))
        text_response = result["content"][0]["text"].strip()

        # Remove ```json fences if present
        text_response = re.sub(r"```json\s*|\s*```", "", text_response).strip()

        final_result = json.loads(text_response)
        print(f"✓ Bedrock combined processing successful for {company_name}")

        # Add the 'is_interesting' flag based on the LLM's 'score'
        final_result["is_interesting"] = final_result.get("score", 0) >= 40

        return final_result

    except Exception as e:
        print(f"✗ Bedrock error for {company_name}: {e}")
        # Fallback to a simple, low-confidence heuristic result on error
        return {
            "has_warehouse": False, "warehouse_type": "unknown", "approx_scale": "unknown",
            "approx_pallet_capacity": None, "industry": "", "is_public_sector": False,
            "safety_focus": False, "website_confidence": 0.1,
            "score": 0, "segment": "", "sales_note": f"Bedrock failed: {str(e)}"
        }


# ========== MAIN LAMBDA HANDLER (Refactored) ==========

def lambda_handler(event, context):
    """
    Main Lambda handler - reads companies.csv from S3,
    fetches websites, uses Bedrock to perform combined extraction and scoring,
    and returns a preview.
    """
    print("=" * 60)
    print(f"Lambda invoked at {datetime.utcnow().isoformat()}Z")
    print("=" * 60)

    print(f"📁 Reading CSV from s3://{BUCKET}/{INPUT_KEY}")

    try:
        obj = s3.get_object(Bucket=BUCKET, Key=INPUT_KEY)
        body = obj["Body"].read().decode("utf-8").splitlines()
        reader = csv.DictReader(body)

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

                # 1. Fetch
                html = fetch_website_html(website) if website else ""
                text = strip_html_tags(html)

                # 2. Extract, Score, and Note in one LLM call
                llm_output = call_bedrock_extract_and_score(name, location, text)

                # Unpack and combine the results
                score = llm_output.get("score", 0)
                segment = llm_output.get("segment", "")
                sales_note = llm_output.get("sales_note", "LLM processing failed or note unavailable.")
                is_interesting = llm_output.get("is_interesting", False)

                emoji = (
                    "🔥" if segment == "A" else
                    "✓" if segment == "B" else
                    "○" if segment == "C" else
                    "✗"
                )
                print(f"   {emoji} Score: {score}/100 | Segment: {segment or 'Not interesting'}")
                print(f"   📝 {sales_note}")

                # Create the final record, including all features from the LLM
                rec = {
                    "company_id": cid,
                    "name": name,
                    "location": location,
                    "website": website,
                    "features": {k: v for k, v in llm_output.items() if k not in ["score", "segment", "sales_note", "is_interesting"]},
                    "score": score,
                    "segment": segment,
                    "is_interesting": is_interesting,
                    "sales_note": sales_note,
                }

                results.append(rec)
                processed += 1

            except Exception as e:
                errors += 1
                print(f"✗ Error processing row: {e}")
                continue

        print("\n" + "=" * 60)
        print("📊 SUMMARY")
        print(f"   Total processed: {processed}")
        print(f"   Errors: {errors}")
        print("=" * 60)

        # Only return a preview to keep response small
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "processed": processed,
                    "errors": errors,
                    "preview": results[:5],  # first 5 companies
                },
                indent=2,
            ),
        }

    except Exception as e:
        print(f"💥 FATAL ERROR: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
        }