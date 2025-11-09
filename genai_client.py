# genai_client.py
import os
from google import genai
from dotenv import load_dotenv

# Load .env file from project root (or current working directory)
# This will populate os.environ with any variables present in .env
load_dotenv(dotenv_path=".env", override=False)

GENIE_MODEL = "gemini-2.5-flash"

def get_api_key_from_env() -> str:
    """
    Tries multiple places to find the API key:
    1. environment variable GENAI_API_KEY (already set by load_dotenv if .env exists)
    2. fallback: OS environment (same)
    If not found, returns empty string.
    """
    return os.getenv("GENAI_API_KEY", "")  # empty string if missing

def create_client(api_key: str | None = None):
    """
    Create genai client. If api_key is omitted, auto-read from environment (.env loaded earlier).
    Will raise an informative error if key is missing or empty.
    """
    key = api_key or get_api_key_from_env()
    if not key:
        raise EnvironmentError(
            "GENAI_API_KEY not found. Set it in a .env file (GENAI_API_KEY=...) or as an environment variable."
        )
    return genai.Client(api_key=key)

def compose_prompt_with_context(question: str, passages: list):
    system_prompt = (
    "You are a data extraction assistant specialized in bank statements. "
    "TASK: Extract all key transaction and summary fields into a structured JSON object, "
    "and generate short financial insights based on the extracted data. "
    "You MUST follow these rules exactly:\n\n"

    "1) **OUTPUT FORMAT (MANDATORY)**: Return ONLY a single, valid JSON object and nothing else (no explanation, no extra text, no Markdown). "
    "If you cannot find any relevant information in the provided CONTEXT, return the string: "
    "\"I don't know based on the provided context.\" (without additional text). \n\n"

    "2) **JSON SCHEMA** (fields and types):\n"
    "{\n"
    "  \"account_info\": {\n"
    "    \"bank_name\": string | null,\n"
    "    \"account_holder_name\": string | null,\n"
    "    \"masked_account_number\": string | null,  # format: \"****1234\" (last 4 digits)\n"
    "    \"statement_month\": string | null,        # format: YYYY-MM or \"Month YYYY\"\n"
    "    \"account_type\": string | null            # \"checking\" | \"savings\" | \"joint\" | null\n"
    "  },\n"
    "  \"summary_values\": {\n"
    "    \"opening_balance\": number | null,        # numeric, in INR (no currency symbol)\n"
    "    \"closing_balance\": number | null,\n"
    "    \"total_credits\": number | null,\n"
    "    \"total_debits\": number | null,\n"
    "    \"average_daily_balance\": number | null,\n"
    "    \"overdraft_count\": integer | null\n"
    "  },\n"
    "  \"insights\": [ string, ... ],              # short one-line human-readable health statements\n"
    "}\n\n"

    "3) **MISSING / UNCERTAIN DATA**: If a particular value cannot be found or is ambiguous in the CONTEXT, set its value to null and do NOT guess. Still include an empty array for its citation in `field_citations` (e.g. []).\n\n"

    "4) **NUMBERS & FORMATTING**: Return numeric values as numbers (no commas, no currency symbol). "
    "Round to 2 decimal places only if the exact value is present as decimals; otherwise keep integers. "
    "For masked account numbers return only last 4 digits prefixed by asterisks e.g. \"****1234\".\n\n"


    "6) **INSIGHTS**: Provide 1–5 concise sentences in the `insights` array that summarize account health (examples: "
    "\"Account maintained < ₹10 000 average balance during October.\", "
    "\"High number of overdrafts (3) this statement month.\"). Keep them short (<120 characters), factual and based only on extracted numbers.\n\n"

    "7) **BE CONSERVATIVE**: Do NOT invent transactions, balances, or dates. Use only the provided CONTEXT passages. "
    "If the CONTEXT contains multiple candidate values that conflict (e.g., two different closing balances), set the field value to null and list the conflicting passage labels in `field_citations`.\n\n"

    "8) **EXAMPLE OUTPUT** (exact JSON structure, formatted here for clarity; your actual response must be compact JSON only):\n"
    "{\n"
    "  \"account_info\": {\n"
    "    \"bank_name\": \"Example Bank\",\n"
    "    \"account_holder_name\": \"A. B. Customer\",\n"
    "    \"masked_account_number\": \"****4321\",\n"
    "    \"statement_month\": \"2025-10\",\n"
    "    \"account_type\": \"checking\"\n"
    "  },\n"
    "  \"summary_values\": {\n"
    "    \"opening_balance\": 5000.00,\n"
    "    \"closing_balance\": 8500.50,\n"
    "    \"total_credits\": 10000.00,\n"
    "    \"total_debits\": 6500.50,\n"
    "    \"average_daily_balance\": 7000.00,\n"
    "    \"overdraft_count\": 0\n"
    "  },\n"
    "  \"insights\": [\n"
    "    \"Account maintained < ₹10 000 average balance during October.\"\n"
    "  ],\n"
    
    "END OF INSTRUCTIONS. Use only the CONTEXT passages provided after these instructions to extract data and build the JSON."
)

    context_text = "\n\n".join([f"[PASSAGE {i+1}]\n{p}" for i, p in enumerate(passages)])
    user_content = f"{system_prompt}\n\nCONTEXT:\n{context_text}\n\nQUESTION: {question}\n\nAnswer:"
    return user_content

def ask_gemini(client, question: str, passages: list, model: str = GENIE_MODEL):
    content = compose_prompt_with_context(question, passages)
    response = client.models.generate_content(model=model, contents=content)
    text = getattr(response, "text", None) or getattr(response, "output_text", None) or str(response)
    return text
