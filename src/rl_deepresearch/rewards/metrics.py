"""
Verifiable metrics for reward computation.

Key insight from research: verifiable rewards = deterministic gradients = stable training.
LLM-as-judge introduces variance, so minimize its usage.
"""

import re
import json
from collections import Counter
from typing import Any

import numpy as np


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text


def compute_exact_match(prediction: str, reference: str) -> float:
    """
    Compute exact match score.

    Binary reward - correct or not.
    """
    pred_norm = normalize_text(prediction)
    ref_norm = normalize_text(reference)
    return 1.0 if pred_norm == ref_norm else 0.0


def compute_word_f1(prediction: str, reference: str) -> float:
    """
    Compute word-level F1 score.

    Key insight: F1 provides partial credit (dense signal), unlike binary EM.
    This enables gradual learning - more correct words = higher reward.
    """
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)

    # Count overlapping tokens
    overlap = sum((pred_counter & ref_counter).values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_rouge(prediction: str, reference: str, n: int = 1) -> dict[str, float]:
    """
    Compute ROUGE-N scores.

    Returns precision, recall, and F1 for n-grams.
    """
    def get_ngrams(tokens: list[str], n: int) -> Counter:
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if len(pred_tokens) < n or len(ref_tokens) < n:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_ngrams = get_ngrams(pred_tokens, n)
    ref_ngrams = get_ngrams(ref_tokens, n)

    overlap = sum((pred_ngrams & ref_ngrams).values())

    if overlap == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = overlap / sum(pred_ngrams.values())
    recall = overlap / sum(ref_ngrams.values())
    f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_information_gain(
    new_content: str,
    existing_knowledge: list[str],
    embedder: Any | None = None
) -> float:
    """
    Compute information gain from new content.

    Key insight: reward for NEW information, not redundant confirmation.
    This forces diversification of search strategy.

    Args:
        new_content: Newly retrieved content
        existing_knowledge: List of previously found content
        embedder: Optional sentence embedder for semantic comparison

    Returns:
        Information gain score [0, 1]
    """
    if not existing_knowledge:
        return 1.0  # First piece of info always has gain

    new_tokens = set(normalize_text(new_content).split())

    if not new_tokens:
        return 0.0

    # Compute overlap with each existing piece
    existing_tokens = set()
    for content in existing_knowledge:
        existing_tokens.update(normalize_text(content).split())

    # New tokens not seen before
    novel_tokens = new_tokens - existing_tokens
    novelty_ratio = len(novel_tokens) / len(new_tokens)

    # If embedder provided, also compute semantic novelty
    if embedder is not None:
        try:
            new_embedding = embedder.encode(new_content)
            existing_embeddings = embedder.encode(existing_knowledge)

            # Max similarity to any existing content
            similarities = np.dot(existing_embeddings, new_embedding)
            max_similarity = np.max(similarities)

            # Semantic novelty
            semantic_novelty = 1.0 - max_similarity

            # Combine lexical and semantic novelty
            return 0.5 * novelty_ratio + 0.5 * semantic_novelty
        except Exception:
            pass

    return novelty_ratio


def check_format_validity(output: str, expected_format: str = "json") -> dict[str, Any]:
    """
    Check if output follows expected format.

    Key insight: format errors are diagnostic signals for policy instability.
    Penalty: -2.0 to -1.0 (worse than wrong answer).

    Returns:
        dict with:
        - valid: bool
        - error_type: str | None
        - parsed: Any | None
    """
    result = {
        "valid": False,
        "error_type": None,
        "parsed": None,
    }

    if expected_format == "json":
        # Try to extract JSON from output
        json_patterns = [
            r'```json\s*(.*?)\s*```',  # Markdown code block
            r'\{[^{}]*\}',  # Simple object
            r'\[[^\[\]]*\]',  # Simple array
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, output, re.DOTALL)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    result["valid"] = True
                    result["parsed"] = parsed
                    return result
                except json.JSONDecodeError:
                    continue

        # Try parsing entire output
        try:
            parsed = json.loads(output.strip())
            result["valid"] = True
            result["parsed"] = parsed
            return result
        except json.JSONDecodeError as e:
            result["error_type"] = "json_parse_error"
            result["error_message"] = str(e)

    elif expected_format == "tool_call":
        # Check for valid tool call format
        tool_patterns = [
            r'<tool>\s*(.*?)\s*</tool>',
            r'\[TOOL\]\s*(.*?)\s*\[/TOOL\]',
            r'```tool\s*(.*?)\s*```',
        ]

        for pattern in tool_patterns:
            match = re.search(pattern, output, re.DOTALL)
            if match:
                tool_content = match.group(1)
                try:
                    parsed = json.loads(tool_content)
                    if "name" in parsed and "arguments" in parsed:
                        result["valid"] = True
                        result["parsed"] = parsed
                        return result
                    else:
                        result["error_type"] = "missing_tool_fields"
                except json.JSONDecodeError:
                    result["error_type"] = "tool_json_error"

        result["error_type"] = result.get("error_type", "no_tool_found")

    elif expected_format == "answer":
        # Check for answer tags
        answer_patterns = [
            r'<answer>\s*(.*?)\s*</answer>',
            r'\[ANSWER\]\s*(.*?)\s*\[/ANSWER\]',
            r'Answer:\s*(.+?)(?:\n|$)',
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)
            if match:
                result["valid"] = True
                result["parsed"] = match.group(1).strip()
                return result

        result["error_type"] = "no_answer_found"

    return result


def check_tool_call_validity(
    tool_call: dict[str, Any],
    available_tools: list[str],
    tool_schemas: dict[str, dict] | None = None
) -> dict[str, Any]:
    """
    Validate tool call against available tools and schemas.

    Key insight: separate penalties for different error types
    - bad_tool_name: -1.5
    - bad_tool_args: -1.0
    """
    result = {
        "valid": False,
        "error_type": None,
        "error_details": None,
    }

    # Check tool name
    tool_name = tool_call.get("name")
    if not tool_name:
        result["error_type"] = "missing_tool_name"
        return result

    if tool_name not in available_tools:
        result["error_type"] = "bad_tool_name"
        result["error_details"] = f"Unknown tool: {tool_name}. Available: {available_tools}"
        return result

    # Check arguments
    args = tool_call.get("arguments", {})

    if tool_schemas and tool_name in tool_schemas:
        schema = tool_schemas[tool_name]
        required_params = schema.get("required", [])

        for param in required_params:
            if param not in args:
                result["error_type"] = "bad_tool_args"
                result["error_details"] = f"Missing required parameter: {param}"
                return result

    result["valid"] = True
    return result


def check_citation_correctness(
    cited_sources: list[str],
    actual_sources: list[str],
    answer: str,
    evidence: list[dict[str, str]]
) -> dict[str, Any]:
    """
    Check if citations are correct and support the answer.

    Key insight: citation correctness as proxy for grounding.
    Correct answer with wrong citation = partial failure.
    This is anti-hallucination mechanism.

    Returns:
        dict with:
        - correct_citations: int
        - total_citations: int
        - precision: float
        - recall: float
        - grounded: bool (answer supported by cited evidence)
    """
    result = {
        "correct_citations": 0,
        "total_citations": len(cited_sources),
        "precision": 0.0,
        "recall": 0.0,
        "grounded": False,
    }

    if not cited_sources:
        return result

    # Normalize source IDs for comparison
    cited_set = set(s.lower().strip() for s in cited_sources)
    actual_set = set(s.lower().strip() for s in actual_sources)

    # Count correct citations
    correct = len(cited_set & actual_set)
    result["correct_citations"] = correct

    # Precision: of cited sources, how many are correct?
    result["precision"] = correct / len(cited_sources) if cited_sources else 0.0

    # Recall: of actual sources, how many were cited?
    result["recall"] = correct / len(actual_sources) if actual_sources else 0.0

    # Check if answer is grounded in evidence
    answer_tokens = set(normalize_text(answer).split())
    evidence_tokens = set()
    for ev in evidence:
        content = ev.get("content", "")
        evidence_tokens.update(normalize_text(content).split())

    # Answer is grounded if significant overlap with evidence
    if answer_tokens and evidence_tokens:
        overlap = len(answer_tokens & evidence_tokens)
        grounding_ratio = overlap / len(answer_tokens)
        result["grounded"] = grounding_ratio > 0.3

    return result


def detect_hallucination_signals(
    answer: str,
    retrieved_content: list[str],
    question: str
) -> dict[str, Any]:
    """
    Detect potential hallucination signals.

    This is a heuristic check - not a replacement for verification.

    Returns indicators that might suggest hallucination:
    - specificity_without_source: specific claims not in retrieved content
    - hedging_language: uncertainty markers (may suggest calibration)
    - confidence_without_evidence: strong claims without support
    """
    result = {
        "specificity_without_source": False,
        "hedging_language": False,
        "confidence_markers": [],
        "unsupported_claims": [],
    }

    # Check for hedging (positive sign of calibration)
    hedging_patterns = [
        r'\b(may|might|could|possibly|perhaps|likely|unlikely|uncertain|unclear)\b',
        r'\b(I\'m not sure|I don\'t know|not certain|hard to say)\b',
        r'\b(according to|based on|the source|the document)\b',
    ]

    for pattern in hedging_patterns:
        if re.search(pattern, answer, re.IGNORECASE):
            result["hedging_language"] = True
            break

    # Check for overly specific claims
    specific_patterns = [
        r'\b\d{4}\b',  # Years
        r'\b\d+(?:\.\d+)?%\b',  # Percentages
        r'\b\d+(?:,\d{3})*(?:\.\d+)?\b',  # Numbers
        r'"[^"]{10,}"',  # Quoted content
    ]

    # Collect all retrieved text
    all_retrieved = " ".join(retrieved_content).lower()

    for pattern in specific_patterns:
        matches = re.findall(pattern, answer)
        for match in matches:
            if match.lower() not in all_retrieved:
                result["unsupported_claims"].append(match)

    result["specificity_without_source"] = len(result["unsupported_claims"]) > 0

    return result


def compute_answer_completeness(
    answer: str,
    question: str,
    expected_answer: str | None = None,
    answer_type: str = "factoid"
) -> float:
    """
    Compute how complete an answer is relative to the question.

    For factoid questions: checking if answer addresses the question
    For multi-part questions: checking if all parts are addressed
    """
    if not answer.strip():
        return 0.0

    # Extract question components
    question_tokens = set(normalize_text(question).split())
    answer_tokens = set(normalize_text(answer).split())

    # Remove common words
    stopwords = {
        'what', 'who', 'where', 'when', 'why', 'how', 'is', 'are', 'was',
        'were', 'the', 'a', 'an', 'of', 'in', 'to', 'for', 'and', 'or'
    }
    question_content = question_tokens - stopwords

    # Check if answer addresses question content
    addressed = len(answer_tokens & question_content) / len(question_content) if question_content else 0

    # If expected answer provided, also check coverage
    if expected_answer:
        expected_tokens = set(normalize_text(expected_answer).split())
        coverage = len(answer_tokens & expected_tokens) / len(expected_tokens) if expected_tokens else 0
        return 0.5 * addressed + 0.5 * coverage

    return addressed
