"""
Prompt templates for research agent.

Key insight: structured prompts improve RL training stability.
Format constraints as implicit curriculum.
"""

from typing import Any


class SystemPrompts:
    """System prompt templates."""

    RESEARCH_AGENT = """You are a research assistant that answers questions by searching and reading documents.

Available tools:
{tools}

Instructions:
1. Search for relevant documents using keyword_search or semantic_search
2. Read promising documents to find evidence
3. When you have enough evidence, use the answer tool with citations
4. If you cannot find enough evidence, use dont_know instead of guessing

Output format:
{{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}

Think carefully before each action. Be efficient and avoid redundant searches."""

    THINKING_AGENT = """You are a research assistant that answers questions by thinking step-by-step.

<think>
Use this section to reason about the problem:
- What do I know?
- What do I need to find out?
- What should I search for?
</think>

Then output your action as JSON:
{{"name": "tool_name", "arguments": {{...}}}}

Available tools:
{tools}

Be thorough but efficient. Cite your sources."""

    MINIMAL = """Research assistant. Answer questions using provided tools.
Tools: {tools}
Format: {{"name": "tool", "arguments": {{...}}}}"""


class PromptBuilder:
    """Build prompts for different scenarios."""

    def __init__(self, style: str = "research"):
        self.style = style

    def build_system_prompt(
        self,
        tools_description: str,
        style: str | None = None,
    ) -> str:
        """Build system prompt."""
        style = style or self.style

        if style == "thinking":
            template = SystemPrompts.THINKING_AGENT
        elif style == "minimal":
            template = SystemPrompts.MINIMAL
        else:
            template = SystemPrompts.RESEARCH_AGENT

        return template.format(tools=tools_description)

    def build_user_prompt(
        self,
        question: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Build user prompt with context."""
        parts = [f"Question: {question}"]

        if context:
            if context.get("searched_queries"):
                parts.append(f"\nPrevious searches: {', '.join(context['searched_queries'][-5:])}")

            if context.get("found_docs"):
                parts.append(f"\nFound documents: {', '.join(context['found_docs'][-10:])}")

            if context.get("knowledge"):
                parts.append("\nKey findings:")
                for note in context["knowledge"][-5:]:
                    parts.append(f"  - {note[:200]}")

            if context.get("last_observation"):
                obs = context["last_observation"]
                if len(obs) > 1500:
                    obs = obs[:1500] + "\n[TRUNCATED]"
                parts.append(f"\nLast result:\n{obs}")

        return "\n".join(parts)

    def build_few_shot_examples(
        self,
        num_examples: int = 2,
    ) -> str:
        """Build few-shot examples for the prompt."""
        examples = [
            {
                "question": "What is the capital of France?",
                "actions": [
                    '{"name": "keyword_search", "arguments": {"query": "capital of France"}}',
                    '{"name": "read_document", "arguments": {"doc_id": "france_123"}}',
                    '{"name": "answer", "arguments": {"answer": "Paris", "sources": ["france_123"]}}',
                ],
            },
            {
                "question": "Who wrote Romeo and Juliet?",
                "actions": [
                    '{"name": "semantic_search", "arguments": {"query": "author Romeo and Juliet play"}}',
                    '{"name": "read_section", "arguments": {"section_id": "shakespeare_001:2"}}',
                    '{"name": "answer", "arguments": {"answer": "William Shakespeare", "sources": ["shakespeare_001"]}}',
                ],
            },
        ]

        selected = examples[:num_examples]
        formatted = []

        for ex in selected:
            formatted.append(f"Question: {ex['question']}")
            for i, action in enumerate(ex['actions'], 1):
                formatted.append(f"Action {i}: {action}")
            formatted.append("")

        return "\n".join(formatted)

    def build_cot_prompt(
        self,
        question: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Build chain-of-thought prompt."""
        user_prompt = self.build_user_prompt(question, context)

        return f"""{user_prompt}

Let me think through this step by step:
1. First, I need to understand what information is being asked for.
2. Then, I should search for relevant documents.
3. After reading, I'll synthesize the answer from the evidence.

Now, let me take my first action:"""


class ToolDescriptionBuilder:
    """Build tool descriptions for prompts."""

    @staticmethod
    def build_compact(tools: list[dict]) -> str:
        """Compact tool descriptions."""
        lines = []
        for tool in tools:
            name = tool["name"]
            desc = tool.get("description", "")[:50]
            params = tool.get("parameters", {}).get("properties", {})
            param_list = ", ".join(params.keys())
            lines.append(f"- {name}({param_list}): {desc}")
        return "\n".join(lines)

    @staticmethod
    def build_detailed(tools: list[dict]) -> str:
        """Detailed tool descriptions with schemas."""
        lines = []
        for tool in tools:
            lines.append(f"### {tool['name']}")
            lines.append(f"{tool.get('description', '')}")
            lines.append("Parameters:")

            params = tool.get("parameters", {}).get("properties", {})
            required = tool.get("parameters", {}).get("required", [])

            for param, schema in params.items():
                req = "(required)" if param in required else "(optional)"
                ptype = schema.get("type", "any")
                pdesc = schema.get("description", "")
                lines.append(f"  - {param} [{ptype}] {req}: {pdesc}")

            lines.append("")

        return "\n".join(lines)
