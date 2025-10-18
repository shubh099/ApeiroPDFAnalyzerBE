from typing import List, Dict, Any
import google.generativeai as genai
import anthropic
import httpx
import os

# Configure Gemini API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

def get_anthropic_client():
    """Get Anthropic client for web search clarifications."""
    # Create httpx client without proxies to avoid compatibility issues
    http_client = httpx.Client()
    return anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        http_client=http_client
    )

def build_service_registry(tables_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Creates a dictionary mapping service names to all their instances across tables.
    """
    service_registry = {}
    for table in tables_data:
        for row in table['rows']:
            # Assuming the service name is in the first column ('Scope')
            service_name = row[0].lower().strip()
            if service_name not in service_registry:
                service_registry[service_name] = []
            
            service_registry[service_name].append({
                "fund": table['fund'],
                "category": table['category'],
                # Extract other attributes based on column headers
                "tariff": row[2] if len(row) > 2 else None,
                "access_rules": row[3] if len(row) > 3 else None,
                "row_data": row
            })
    return service_registry

import json

def format_instances_for_prompt(instances: List[Dict[str, Any]]) -> str:
    """Formats a list of service instances for inclusion in a prompt."""
    formatted_instances = []
    for i, instance in enumerate(instances, 1):
        formatted_instances.append(f"Instance {i}:\n" + json.dumps(instance, indent=2))
    return "\n".join(formatted_instances)

def analyze_contradiction(service_name: str, instances: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Use Claude to determine if instances represent a true contradiction.
    """
    
    prompt = f"""You are analyzing healthcare insurance policy contradictions.

Service: "{service_name}"

Instances found across different packages:
{format_instances_for_prompt(instances)}

Task: Analyze if there are contradictions, ambiguities, or conflicts between these instances.

Consider:
- Same service with different tariffs at same facility level
- Conflicting eligibility criteria (age, frequency, etc.)
- Coverage conflicts (included in one, excluded in another)
- Ambiguous or unclear rules that could cause confusion

Return ONLY a valid JSON object (no markdown, no explanation):
{{
  "has_contradiction": true/false,
  "severity": "high"/"medium"/"low",
  "issue": "Brief description of the contradiction",
  "evidence": ["Quote from instance 1", "Quote from instance 2"],
  "impact": "Practical consequences for patients/providers"
}}

If no contradiction exists, set has_contradiction to false and provide minimal other fields.
"""

    model = genai.GenerativeModel("models/gemini-2.5-pro")
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.2,  # Slightly increased for more nuanced results
            "max_output_tokens": 2000,
            "response_mime_type": "application/json"
        }
    )

    return json.loads(response.text)

def detect_contradictions(tables_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Detects contradictions in the extracted table data.
    """
    service_registry = build_service_registry(tables_data)
    potential_contradictions = {
        service: instances
        for service, instances in service_registry.items()
        if len(instances) > 1
    }
    
    contradictions = []
    for service, instances in potential_contradictions.items():
        analysis = analyze_contradiction(service, instances)
        
        if analysis['has_contradiction']:
            contradictions.append({
                'type': 'CONTRADICTION',
                'service': service,
                'severity': analysis['severity'],
                'locations': [f"{inst['fund']} > {inst['category']}" for inst in instances],
                'issue': analysis['issue'],
                'evidence': analysis['evidence'],
                'impact': analysis['impact']
            })

    return contradictions

def extract_entities(tables_data: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Extract:
    - Conditions/diseases mentioned in Scope columns
    - Services/procedures listed
    - Undefined terms (services mentioned without clear definition or tariff)
    """
    conditions = []
    services = []
    undefined_terms = []

    for table in tables_data:
        for row in table['rows']:
            # Extract conditions and services from the 'Scope' column
            scope_text = row[0].lower()
            # (Add more sophisticated entity extraction logic here)
            if "cancer" in scope_text:
                conditions.append("cancer")
            if "dialysis" in scope_text:
                services.append("dialysis")
            
            # Check for undefined terms
            if "basic radiological examinations" in scope_text and not row[2]:
                undefined_terms.append("basic radiological examinations")

    return {
        "conditions": list(set(conditions)),
        "services": list(set(services)),
        "undefined_terms": list(set(undefined_terms))
    }

def format_tables_summary(tables_data: List[Dict[str, Any]]) -> str:
    """Formats a summary of the tables for inclusion in a prompt."""
    summary = []
    for table in tables_data:
        summary.append(f"- Fund: {table['fund']}, Category: {table['category']}")
    return "\n".join(summary)

def analyze_gaps(entities: Dict[str, List[str]], tables_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Use Claude to identify coverage gaps.
    """
    
    prompt = f"""You are analyzing healthcare insurance coverage for gaps and ambiguities.

Extracted Data:
- Conditions mentioned: {entities['conditions']}
- Services available: {entities['services']}
- Undefined terms: {entities['undefined_terms']}

Full context:
{format_tables_summary(tables_data)}

Task: Identify gaps where:
1. A condition is mentioned but has no clear treatment path or coverage
2. A service is mentioned but lacks definition or pricing
3. Terms are used without clear definition, creating ambiguity for patients/providers

Return ONLY a valid JSON array (no markdown):
[
  {{
    "type": "GAP",
    "severity": "high"/"medium"/"low",
    "category": "missing_treatment"/"undefined_service"/"ambiguous_term",
    "issue": "Brief description",
    "evidence": "Specific quotes or references",
    "impact": "Practical consequences",
    "location": "Fund > Category (if applicable)"
  }}
]

Return empty array [] if no significant gaps found.
"""

    model = genai.GenerativeModel("models/gemini-2.5-pro")
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.2,  # Slightly increased for more nuanced results
            "max_output_tokens": 3000,
            "response_mime_type": "application/json"
        }
    )

    return json.loads(response.text)

def find_gaps(tables_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Finds gaps in the extracted table data.
    """
    entities = extract_entities(tables_data)
    gaps = analyze_gaps(entities, tables_data)
    return gaps

def get_clarification(finding: Dict[str, Any], pdf_context: str) -> str:
    """
    Use Claude with web_search to find clarifications.

    Args:
        finding: The contradiction or gap finding
        pdf_context: The document context string extracted by Gemini
    """

    # The pdf_context is now the document_context extracted by Gemini
    system_context = pdf_context

    # Construct prompt
    prompt = f"""You are helping clarify healthcare insurance policy issues.

System Context: {system_context}

Finding Details:
- Type: {finding['type']}
- Issue: {finding['issue']}
- Evidence: {finding['evidence']}
- Impact: {finding['impact']}

Task:
1. Search for official guidelines, standards, or regulations related to this issue
2. Prioritize official government health ministry sites, WHO guidelines, and healthcare standards
3. Provide an INFORMATIVE clarification (not prescriptive - don't tell them what to do)
4. Cite your sources with URLs when available

Use web search to find relevant, up-to-date information. Return a clear, professional clarification that helps understand the issue better.
"""

    client = get_anthropic_client()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        tools=[{
            "type": "web_search_20250305",
            "name": "web_search"
        }],
        messages=[{"role": "user", "content": prompt}]
    )

    # Extract text from response (handle tool use if present)
    clarification_text = ""
    for block in response.content:
        if block.type == "text":
            clarification_text += block.text

    return clarification_text