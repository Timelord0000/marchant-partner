"""
LLM Orchestration Layer — provider-agnostic wrapper for Gemini and Groq.
Produces structured recommendations combining regression signal, merchant profile, and roadmap.
"""

import json
import os
from typing import Any, Optional

# Lazy imports — only loaded when called
_gemini_client = None
_groq_client = None


def _get_gemini():
    global _gemini_client
    if _gemini_client is None:
        import google.generativeai as genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set in environment. Add it to .env file.")
        genai.configure(api_key=api_key)
        _gemini_client = genai.GenerativeModel("gemini-2.0-flash")
    return _gemini_client


def _get_groq():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set in environment. Add it to .env file.")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


SYSTEM_PROMPT = """You are a business growth advisor for small merchants in India.
You combine statistical signals, merchant-specific patterns, and a proven growth playbook
to give actionable, specific advice.

RULES:
1. You MUST cite a specific roadmap stage in every recommendation.
2. You MUST cite a specific data signal (feature contribution) that triggered the advice.
3. Never invent statistics or percentages not present in the input data.
4. Keep advice practical — these are small shop owners, not MBAs.
5. Speak in simple, clear language. Use ₹ for currency.

OUTPUT: Return ONLY valid JSON with this exact structure:
{
  "recommendation": "The specific, actionable suggestion",
  "cited_signal": "Which regression feature contributed most and why it matters",
  "cited_roadmap_stage": "The stage ID from the roadmap (e.g. stage_2_retention)",
  "explanation": "Plain-language explanation of why this advice fits this merchant right now"
}
"""


def _build_prompt(
    regression_signal: dict[str, Any],
    merchant_profile: str,
    roadmap_chunks: list[dict[str, Any]],
) -> str:
    """Constructs the LLM prompt from all three input layers."""
    # Format regression signal
    signal_text = (
        f"Predicted annual revenue growth: {regression_signal['predicted_growth_percent']}%\n"
        f"Model confidence (R²): {regression_signal['confidence']}\n"
        f"Feature contributions:\n"
    )
    for feat, contrib in sorted(
        regression_signal["feature_contributions"].items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    ):
        signal_text += f"  - {feat}: {contrib:+.4f}\n"

    # Format roadmap chunks
    roadmap_text = ""
    for chunk in roadmap_chunks:
        strategies = "\n".join(
            f"  - {s['title']}: {s['detail']}" for s in chunk["strategies"]
        )
        roadmap_text += (
            f"\n## {chunk['id']} — {chunk['name']}\n"
            f"{chunk['description']}\n"
            f"Strategies:\n{strategies}\n"
        )

    return f"""Based on the following three inputs, give ONE specific growth recommendation.

=== REGRESSION SIGNAL ===
{signal_text}

=== MERCHANT PROFILE ===
{merchant_profile}

=== GROWTH ROADMAP STAGES ===
{roadmap_text}

Give your recommendation as JSON only. No markdown, no extra text."""


def _parse_response(text: str) -> dict[str, Any]:
    """Extracts structured JSON from LLM response, handling markdown fences."""
    cleaned = text.strip()
    # Remove markdown code fences if present
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last lines (fences)
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON in the response
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(cleaned[start:end])
        raise ValueError(f"Could not parse JSON from LLM response: {cleaned[:200]}")


def _validate_output(data: dict[str, Any]) -> bool:
    """Validates the output contract."""
    required_keys = ["recommendation", "cited_signal", "cited_roadmap_stage", "explanation"]
    return all(k in data and isinstance(data[k], str) and len(data[k]) > 0 for k in required_keys)


def generate_recommendation(
    regression_signal: dict[str, Any],
    merchant_profile: str,
    roadmap_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Primary entry point: generates a structured recommendation using Gemini (primary)
    with Groq as fallback.

    Args:
        regression_signal: output from RegressionEngine.predict()
        merchant_profile: text summary from MerchantStore.get_profile_summary()
        roadmap_chunks: list of roadmap stage dicts from RoadmapRetriever.retrieve()

    Returns:
        dict with keys: recommendation, cited_signal, cited_roadmap_stage, explanation
    """
    prompt = _build_prompt(regression_signal, merchant_profile, roadmap_chunks)

    # Try Gemini first
    try:
        model = _get_gemini()
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.4, "max_output_tokens": 1024},
        )
        result = _parse_response(response.text)
        if _validate_output(result):
            result["_provider"] = "gemini"
            return result
        raise ValueError("Invalid output structure from Gemini")
    except Exception as gemini_err:
        print(f"Gemini failed: {gemini_err}. Falling back to Groq...")

    # Fallback to Groq
    try:
        client = _get_groq()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=1024,
        )
        result = _parse_response(response.choices[0].message.content)
        if _validate_output(result):
            result["_provider"] = "groq"
            return result
        raise ValueError("Invalid output structure from Groq")
    except Exception as groq_err:
        print(f"Groq also failed: {groq_err}")
        return {
            "recommendation": "Unable to generate recommendation at this time. Please try again.",
            "cited_signal": "N/A",
            "cited_roadmap_stage": "N/A",
            "explanation": f"Both LLM providers failed. Gemini: {gemini_err}. Groq: {groq_err}.",
            "_provider": "none",
        }
