import base64
import json
import logging
import os
from typing import List, Dict, Any
import google.generativeai as genai
from pathlib import Path
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# Configure Gemini API
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please add it to your .env file.")
genai.configure(api_key=api_key)


EXTRACTION_PROMPT = """You will be given a PDF document containing benefit package tariffs. Your task is to extract specific tables from this document and format them into a single, valid JSON object.

**Instructions:**

1.  **Extraction Schema:** The tables you must extract follow a specific nested structure:
    * They are grouped under **major headings** (e.g., \"PRIMARY HEALTHCARE FUND\", \"SOCIAL HEALTH INSURANCE FUND\"). These will be the parent topics.
    * Within each major heading, there are **sub-topic tables** (e.g., \"OUTPATIENT CARE SERVICES\", \"RENAL CARE PACKAGE\").
    * These sub-topic tables always have four columns: \"Scope\", \"Access Point\", \"Tariff\", and \"Access Rules\".

2.  **Exclusion Rule (Crucial):**
    * You **MUST STOP** processing when you reach the heading \"ANNEX 1 - SURGICAL PACKAGE\".
    * Do **NOT** include any data from \"ANNEX 1\" or any content that follows it.

3.  **JSON Format:**
    * The final output MUST be a single JSON object.
    * The structure must be: `{\"PARENT_TOPIC\": [{\"CHILD_SUBTOPIC\": {\"Scope\": [], \"Access Point\": [], \"Tariff\": [], \"Access Rules\": []}}]}`
    * Collect all text from a single cell into one string, even if it has multiple lines.

Do not include any other text or explanations in your response outside of the final, valid JSON object.

**Example of the final JSON structure:**

```json
{
  \"PRIMARY HEALTHCARE FUND\": [
    {
      \"OUTPATIENT CARE SERVICES\": {
        \"Scope\": [\"data...\", \"data...\"],
        \"Access Point\": [\"data...\"],
        \"Tariff\": [\"data...\", \"data...\"],
        \"Access Rules\": [\"data...\", \"data...\"]
      }
    },
    {
      \"MATERNITY, NEWBORN AND CHILD HEALTH SERVICES\": {
        \"Scope\": [\"data...\"],
        \"Access Point\": [\"data...\"],
        \"Tariff\": [\"data...\"],
        \"Access Rules\": [\"data...\"]
      }
    }
  ],
  \"SOCIAL HEALTH INSURANCE FUND\": [
    {
      \"OUTPATIENT CARE SERVICES\": {
        \"Scope\": [\"data...\"],
        \"Access Point\": [\"data...\"],
        \"Tariff\": [\"data...\"],
        \"Access Rules\": [\"data...\"]
      }
    }
  ],
  \"EMERGENCY, CHRONIC AND CRITICAL ILLNESS FUND\": [
    {
      \"AMBULANCE EVACUATION SERVICES\": {
        \"Scope\": [\"data...\"],
        \"Access Point\": [\"data...\"],
        \"Tariff\": [\"data...\"],
        \"Access Rules\": [\"data...\"]
      }
    }
  ]
}
```"""  # noqa: E501

def extract_tables_with_gemini(pdf_path: str) -> Dict[str, Any]:
    """
    Extract tables from PDF using Gemini 2.0 Flash with inline base64 encoding.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        JSON object with extracted tables
    """
    logger.info(f"\n{'='*60}")
    logger.info("Starting Gemini PDF Table Extraction")
    logger.info(f"{'='*60}")
    logger.info(f"PDF Path: {pdf_path}")

    try:
        # Read PDF file and encode as base64
        logger.info("\nReading and encoding PDF file...")
        with open(pdf_path, 'rb') as f:
            pdf_data = f.read()

        pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
        logger.info(f"✓ PDF file encoded successfully ({len(pdf_data)} bytes)")

        # Use Gemini 2.5 Pro model (latest available)
        model = genai.GenerativeModel("models/gemini-2.5-pro")
        logger.info(f"✓ Using model: models/gemini-2.5-pro")

        # Create multimodal content with text prompt and inline PDF data
        logger.info("\nSending extraction request to Gemini...")
        logger.info("This may take a moment for large PDFs...\n")

        # Create the multimodal request
        content_parts = [
            {"text": EXTRACTION_PROMPT},
            {
                "inline_data": {
                    "mime_type": "application/pdf",
                    "data": pdf_base64
                }
            }
        ]

        response = model.generate_content(
            content_parts,
            generation_config={
                "temperature": 0.1,  # Low temperature for more deterministic output
                "max_output_tokens": 16384,  # Allow for large JSON responses
                "response_mime_type": "application/json"  # Force JSON output
            },
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )

        logger.info("✓ Response received from Gemini")

        # Extract the JSON from the response
        # With response_mime_type="application/json", we get pure JSON (no markdown)
        response_text = response.text.strip()

        # Log the raw response for debugging
        logger.info(f"\nRaw Gemini Response:\n{response_text}\n")

        # Parse JSON directly (no markdown cleanup needed)
        logger.info("\nParsing JSON response...")
        extracted_data = json.loads(response_text)
        logger.info("✓ JSON parsed successfully")

        # Log summary
        logger.info(f"\n{'='*60}")
        logger.info("EXTRACTION SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Top-level keys found: {list(extracted_data.keys())}")

        for key in extracted_data.keys():
            if isinstance(extracted_data[key], list):
                logger.info(f"  - {key}: {len(extracted_data[key])} subtopics")
            elif isinstance(extracted_data[key], dict):
                logger.info(f"  - {key}: Direct table with {len(extracted_data[key])} columns")

        logger.info(f"{'='*60}\n")

        return extracted_data

    except json.JSONDecodeError as e:
        logger.error(f"\n❌ JSON Parse Error: {e}")
        logger.error(f"Response text: {response_text[:500]}...")
        raise Exception(f"Failed to parse Gemini response as JSON: {e}")

    except Exception as e:
        logger.error(f"\n❌ Gemini Extraction Error: {e}")
        raise Exception(f"Failed to extract tables with Gemini: {e}")

def transform_gemini_output_to_frontend_format(gemini_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Transform Gemini's JSON output to the format expected by the frontend.

    Gemini format:
    {
      "PRIMARY HEALTHCARE FUND": [
        {
          "OUTPATIENT CARE SERVICES": {
            "Scope": ["data1", "data2"],
            "Access Point": ["data1", "data2"],
            "Tariff": ["data1", "data2"],
            "Access Rules": ["data1", "data2"]
          }
        }
      ],
      "ANNEX 1 - SURGICAL PACKAGE": {
        "Specialty": ["data1", "data2"],
        "Intervention": ["data1", "data2"],
        "Tariff": ["data1", "data2"]
      }
    }

    Frontend format:
    [
      {
        "fund": "PRIMARY HEALTHCARE FUND",
        "category": "OUTPATIENT CARE SERVICES",
        "headers": ["Scope", "Access Point", "Tariff", "Access Rules"],
        "rows": [
          ["data1", "data1", "data1", "data1"],
          ["data2", "data2", "data2", "data2"]
        ]
      },
      {
        "fund": "ANNEX 1 - SURGICAL PACKAGE",
        "category": "ANNEX 1 - SURGICAL PACKAGE",
        "headers": ["Specialty", "Intervention", "Tariff"],
        "rows": [
          ["data1", "data1", "data1"],
          ["data2", "data2", "data2"]
        ]
      }
    ]
    """
    logger.info("\n{'='*60}")
    logger.info("Transforming Gemini output to frontend format")
    logger.info(f"{'='*60}\n")

    transformed_tables = []

    for key, value in gemini_data.items():
        # Case 1: Main Benefit Tables (list of subtopics)
        if isinstance(value, list):
            fund = key
            logger.info(f"Processing fund: {fund}")

            for subtopic_dict in value:
                for category, table_data in subtopic_dict.items():
                    logger.info(f"  - Processing category: {category}")

                    # Get headers
                    headers = list(table_data.keys())

                    # Get number of rows (all columns should have same length)
                    num_rows = len(table_data[headers[0]]) if headers else 0

                    # Transform columns to rows
                    rows = []
                    for i in range(num_rows):
                        row = [table_data[header][i] if i < len(table_data[header]) else "" for header in headers]
                        rows.append(row)

                    transformed_tables.append({
                        "fund": fund,
                        "category": category,
                        "headers": headers,
                        "rows": rows
                    })

                    logger.info(f"    ✓ Added table: {len(rows)} rows, {len(headers)} columns")

        # Case 2: Annex Tables (direct dictionary)
        elif isinstance(value, dict):
            fund = key
            category = key  # For annex, fund and category are the same
            logger.info(f"Processing annex table: {fund}")

            # Get headers
            headers = list(value.keys())

            # Get number of rows
            num_rows = len(value[headers[0]]) if headers else 0

            # Transform columns to rows
            rows = []
            for i in range(num_rows):
                row = [value[header][i] if i < len(value[header]) else "" for header in headers]
                rows.append(row)

            transformed_tables.append({
                "fund": fund,
                "category": category,
                "headers": headers,
                "rows": rows
            })

            logger.info(f"  ✓ Added annex table: {len(rows)} rows, {len(headers)} columns")

    logger.info(f"\n{'='*60}")
    logger.info(f"Transformation complete: {len(transformed_tables)} tables created")
    logger.info(f"{'='*60}\n")

    return {
        "document_context": gemini_data.get("document_context", ""),
        "tables": transformed_tables
    }

def extract_tables_from_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Main entry point for table extraction.

    Uses Gemini 2.5 Pro to extract tables from PDF and transforms
    the output to match the frontend's expected format.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        A dictionary containing the document context and the list of tables.
    """
    # Step 1: Extract with Gemini
    gemini_data = extract_tables_with_gemini(pdf_path)

    # Step 2: Transform to frontend format
    frontend_data = transform_gemini_output_to_frontend_format(gemini_data)

    return frontend_data

def get_extraction_summary(tables: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a summary of the extraction results.

    Returns:
    {
        "total_tables": int,
        "total_rows": int,
        "unique_funds": int,
        "unique_categories": int,
        "funds": [str],
        "categories": [str]
    }
    """
    total_tables = len(tables)
    total_rows = sum(len(table.get("rows", [])) for table in tables)
    funds = set(table["fund"] for table in tables if table.get("fund"))
    categories = set(table["category"] for table in tables if table.get("category"))

    return {
        "total_tables": total_tables,
        "total_rows": total_rows,
        "unique_funds": len(funds),
        "unique_categories": len(categories),
        "funds": list(funds),
        "categories": list(categories)
    }