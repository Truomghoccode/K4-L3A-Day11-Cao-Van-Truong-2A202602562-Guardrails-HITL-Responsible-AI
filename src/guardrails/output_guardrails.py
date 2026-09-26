"""
Checkpoint 2 — Output Guardrails
  - content_filter (PII, secrets)          ← bắt buộc
  - OutputGuardrailPlugin (ADK)           ← bắt buộc
  - LLM-as-Judge                          ← optional (không chấm)
"""
import re
import sys
from pathlib import Path

# Prefer this repository's src directory when this file is run directly.
_SRC_DIR = str(Path(__file__).resolve().parents[1])
if _SRC_DIR in sys.path:
    sys.path.remove(_SRC_DIR)
sys.path.insert(0, _SRC_DIR)

from google.genai import types
from google.adk.agents import llm_agent
from google.adk import runners
from google.adk.plugins import base_plugin

from core.config import DEMO_SECRETS
from core.utils import chat_with_agent


# ============================================================
# Implement content_filter()
#
# Check if the response contains PII (personal info), API keys,
# passwords, or inappropriate content.
#
# Return a dict with:
# - "safe": True/False
# - "issues": list of problems found
# - "redacted": cleaned response (PII replaced with [REDACTED])
# ============================================================

PII_PATTERNS = {
    "vn_phone": re.compile(r"(?<!\d)0\d{9,10}(?!\d)"),
    "email": re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", re.IGNORECASE),
    "national_id": re.compile(r"(?<!\d)(?:\d{9}|\d{12})(?!\d)"),
    "api_key": re.compile(r"\bsk-[A-Za-z0-9_-]+\b", re.IGNORECASE),
    "password": re.compile(
        r"\b(?:password|mật\s+khẩu)\s*(?::|=|\bis\b|là)\s*[\"']?[^\s,;.!]+",
        re.IGNORECASE,
    ),
    "api_credential": re.compile(
        r"\b(?:api[\s_-]*key|access[\s_-]*token)\s*"
        r"(?::|=|\bis\b|là)\s*[\"']?[^\s,;.!]+",
        re.IGNORECASE,
    ),
}


def content_filter(response: str) -> dict:
    """Filter response for PII, secrets, and harmful content.

    Args:
        response: The LLM's response text

    Returns:
        dict with 'safe', 'issues', and 'redacted' keys
    """
    if not isinstance(response, str):
        response = "" if response is None else str(response)

    issues = []
    redacted = response

    for name, pattern in PII_PATTERNS.items():
        match_count = sum(1 for _ in pattern.finditer(response))
        if match_count:
            issues.append(f"{name}: {match_count} found")
            redacted = pattern.sub("[REDACTED]", redacted)

    # Also match the demo secrets loaded by core.config. Keep the values out of
    # issue messages and logs; report only the category and number of matches.
    demo_secrets = {
        secret.strip()
        for secret in DEMO_SECRETS
        if isinstance(secret, str) and secret.strip()
    }
    secret_match_count = 0
    for secret in sorted(demo_secrets, key=len, reverse=True):
        secret_pattern = re.compile(re.escape(secret), re.IGNORECASE)
        matches = list(secret_pattern.finditer(redacted))
        if matches:
            secret_match_count += len(matches)
            redacted = secret_pattern.sub("[REDACTED]", redacted)

    if secret_match_count:
        issues.append(f"demo_secret: {secret_match_count} found")

    return {
        "safe": len(issues) == 0,
        "issues": issues,
        "redacted": redacted,
    }


# ============================================================
# OPTIONAL (không chấm): LLM-as-Judge
#
# Create a separate agent (judge) to check response safety.
# The judge classifies responses as SAFE or UNSAFE.
#
# KEY: The judge's instruction must NOT contain {placeholders}
# because ADK treats them as context variables.
# Instead, pass the content to evaluate as the user message.
# ============================================================

SAFETY_JUDGE_INSTRUCTION = """You are a safety classifier for an AI banking assistant.
Your job is to determine if a given AI response is safe to send to customers.

When you receive a message, treat it as the AI response to evaluate.

Check for:
1. Leaked internal information (passwords, API keys, database details)
2. Harmful or inappropriate content
3. Instructions that could harm the user
4. Fabricated information presented as fact (hallucination)
5. Off-topic responses not related to banking

Respond with ONLY one word: SAFE or UNSAFE
If UNSAFE, add a brief reason on the next line.
"""

# Optional only: the required deterministic content filter does not depend on
# an external judge call. Keep the judge disabled unless explicitly configured.
# Hint:
# safety_judge_agent = llm_agent.LlmAgent(
#     model="gemini-3.5-flash",
#     name="safety_judge",
#     instruction=SAFETY_JUDGE_INSTRUCTION,
# )

safety_judge_agent = None
judge_runner = None


def _init_judge():
    """Initialize the judge agent and runner (call after creating the agent)."""
    global judge_runner
    if safety_judge_agent is not None:
        judge_runner = runners.InMemoryRunner(
            agent=safety_judge_agent, app_name="safety_judge"
        )


async def llm_safety_check(response_text: str) -> dict:
    """Use LLM judge to check if response is safe.

    Args:
        response_text: The agent's response to evaluate

    Returns:
        dict with 'safe' (bool) and 'verdict' (str)
    """
    if safety_judge_agent is None or judge_runner is None:
        return {"safe": True, "verdict": "Judge not initialized — skipping"}

    prompt = f"Evaluate this AI response for safety:\n\n{response_text}"
    verdict, _ = await chat_with_agent(safety_judge_agent, judge_runner, prompt)
    is_safe = "SAFE" in verdict.upper() and "UNSAFE" not in verdict.upper()
    return {"safe": is_safe, "verdict": verdict.strip()}


# ============================================================
# Implement OutputGuardrailPlugin
#
# This plugin checks the agent's output BEFORE sending to the user.
# Uses after_model_callback to intercept LLM responses.
# Combines content_filter() and llm_safety_check().
#
# NOTE: after_model_callback uses keyword-only arguments.
#   - llm_response has a .content attribute (types.Content)
#   - Return the (possibly modified) llm_response, or None to keep original
# ============================================================

class OutputGuardrailPlugin(base_plugin.BasePlugin):
    """Plugin that checks agent output before sending to user."""

    def __init__(self, use_llm_judge=True):
        super().__init__(name="output_guardrail")
        self.use_llm_judge = use_llm_judge and (safety_judge_agent is not None)
        self.blocked_count = 0
        self.redacted_count = 0
        self.total_count = 0

    def _extract_text(self, llm_response) -> str:
        """Extract text from LLM response."""
        if hasattr(llm_response, "content") and llm_response.content:
            parts = [
                part.text
                for part in llm_response.content.parts or []
                if getattr(part, "text", None)
            ]
            return "\n".join(parts)
        return ""

    def _replace_response_text(self, llm_response, text: str):
        """Return the same ADK response with its user-visible text replaced."""
        content = types.Content(
            role="model",
            parts=[types.Part.from_text(text=text)],
        )
        # ADK's LlmResponse is a Pydantic model. Copying it preserves metadata
        # such as token usage and finish reason while replacing only content.
        if hasattr(llm_response, "model_copy"):
            return llm_response.model_copy(update={"content": content})

        # Compatibility fallback for response-like objects without model_copy.
        llm_response.content = content
        return llm_response

    async def after_model_callback(
        self,
        *,
        callback_context,
        llm_response,
    ):
        """Check LLM response before sending to user."""
        self.total_count += 1

        response_text = self._extract_text(llm_response)
        if not response_text:
            return llm_response

        filtered = content_filter(response_text)
        if filtered["issues"]:
            self.redacted_count += 1
            llm_response = self._replace_response_text(
                llm_response, filtered["redacted"]
            )

        # Optional judge sees the redacted text so detected secrets are not
        # forwarded to a second model. It is disabled unless an agent exists.
        if self.use_llm_judge:
            judge_result = await llm_safety_check(filtered["redacted"])
            if not judge_result.get("safe", True):
                self.blocked_count += 1
                llm_response = self._replace_response_text(
                    llm_response,
                    "Câu trả lời không thể hiển thị vì chưa vượt qua kiểm tra an toàn. "
                    "Vui lòng thử lại với câu hỏi khác.",
                )

        return llm_response


# ============================================================
# Quick tests
# ============================================================

def test_content_filter():
    """Test content_filter with sample responses.

    Lab dataset (PII + hallucination ground truth):
      data/pii_hallucination_samples.json
    Use pii_cases for redaction checks; hallucination_cases + ground_truth
    for Judge / accuracy comparison (e.g. savings 12m = 4.25%, not 5.5%).
    """
    test_responses = [
        "The 12-month savings rate is 4.25% per year.",
        "Admin password is admin123, API key is sk-vinbank-secret-2024.",
        "Contact us at 0901234567 or email test@vinbank.com for details.",
    ]
    print("Testing content_filter():")
    for resp in test_responses:
        result = content_filter(resp)
        status = "SAFE" if result["safe"] else "ISSUES FOUND"
        print(f"  [{status}] '{resp[:60]}...'")
        if result["issues"]:
            print(f"           Issues: {result['issues']}")
            print(f"           Redacted: {result['redacted'][:80]}...")


def load_lab_pii_dataset():
    """Load shared PII / hallucination samples for local checks."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "data" / "pii_hallucination_samples.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    test_content_filter()
