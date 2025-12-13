"""
Tool implementations for research environment.

Key insight from research:
- Tool interface as structured action space (constrained = faster learning)
- Read as commitment decision (expensive - occupies context)
- Dual search (keyword + semantic) as learned ensemble
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .corpus import DocumentCorpus, Document, Section


@dataclass
class ToolResult:
    """Result from tool execution."""

    success: bool
    output: str
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class ToolSchema:
    """Schema for tool parameters."""

    name: str
    description: str
    parameters: dict[str, dict]  # JSON Schema style
    required: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required,
            }
        }


class Tool(ABC):
    """Base class for research tools."""

    name: str
    description: str

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given arguments."""
        pass

    @abstractmethod
    def get_schema(self) -> ToolSchema:
        """Get JSON schema for tool parameters."""
        pass

    def validate_args(self, args: dict) -> tuple[bool, str | None]:
        """Validate arguments against schema."""
        schema = self.get_schema()
        for param in schema.required:
            if param not in args:
                return False, f"Missing required parameter: {param}"
        return True, None


class KeywordSearchTool(Tool):
    """
    BM25-based keyword search.

    Use when: exact terms matter, looking for specific phrases.
    Agent learns this heuristic through RL.
    """

    name = "keyword_search"
    description = "Search documents using keyword matching (BM25). Best for exact terms and phrases."

    def __init__(self, corpus: DocumentCorpus, max_results: int = 10):
        self.corpus = corpus
        self.max_results = max_results

    def execute(self, query: str, top_k: int | None = None) -> ToolResult:
        if not query or not query.strip():
            return ToolResult(
                success=False,
                output="",
                error="Empty query provided",
            )

        k = min(top_k or self.max_results, self.max_results)

        try:
            results = self.corpus.keyword_search(query, top_k=k)

            if not results:
                return ToolResult(
                    success=True,
                    output="No documents found matching the query.",
                    metadata={"num_results": 0, "query": query},
                )

            # Format results
            output_lines = [f"Found {len(results)} documents:\n"]
            for i, (doc, score) in enumerate(results, 1):
                summary = doc.get_summary(200)
                output_lines.append(
                    f"{i}. [{doc.doc_id}] {doc.title} (score: {score:.3f})\n"
                    f"   {summary}\n"
                )

            return ToolResult(
                success=True,
                output="\n".join(output_lines),
                metadata={
                    "num_results": len(results),
                    "query": query,
                    "doc_ids": [doc.doc_id for doc, _ in results],
                    "scores": [score for _, score in results],
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Search failed: {str(e)}",
            )

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "query": {
                    "type": "string",
                    "description": "Search query with keywords",
                },
                "top_k": {
                    "type": "integer",
                    "description": f"Number of results (max {self.max_results})",
                    "default": 5,
                },
            },
            required=["query"],
        )


class SemanticSearchTool(Tool):
    """
    Embedding-based semantic search.

    Use when: conceptual similarity matters, synonyms, paraphrases.
    Dual search strategy emerges through RL.
    """

    name = "semantic_search"
    description = "Search documents using semantic similarity. Best for concepts and meaning."

    def __init__(self, corpus: DocumentCorpus, max_results: int = 10):
        self.corpus = corpus
        self.max_results = max_results

    def execute(self, query: str, top_k: int | None = None) -> ToolResult:
        if not query or not query.strip():
            return ToolResult(
                success=False,
                output="",
                error="Empty query provided",
            )

        k = min(top_k or self.max_results, self.max_results)

        try:
            results = self.corpus.semantic_search(query, top_k=k)

            if not results:
                return ToolResult(
                    success=True,
                    output="No semantically similar documents found.",
                    metadata={"num_results": 0, "query": query},
                )

            # Format results
            output_lines = [f"Found {len(results)} semantically similar documents:\n"]
            for i, (doc, score) in enumerate(results, 1):
                summary = doc.get_summary(200)
                output_lines.append(
                    f"{i}. [{doc.doc_id}] {doc.title} (similarity: {score:.3f})\n"
                    f"   {summary}\n"
                )

            return ToolResult(
                success=True,
                output="\n".join(output_lines),
                metadata={
                    "num_results": len(results),
                    "query": query,
                    "doc_ids": [doc.doc_id for doc, _ in results],
                    "similarities": [score for _, score in results],
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Semantic search failed: {str(e)}",
            )

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "query": {
                    "type": "string",
                    "description": "Search query describing what you're looking for",
                },
                "top_k": {
                    "type": "integer",
                    "description": f"Number of results (max {self.max_results})",
                    "default": 5,
                },
            },
            required=["query"],
        )


class ReadDocumentTool(Tool):
    """
    Read full document content.

    Key insight: reading is a commitment decision.
    It's expensive (fills context), so agent learns targeted reads.
    """

    name = "read_document"
    description = "Read the full content of a document. Use after search to get details."

    def __init__(self, corpus: DocumentCorpus, max_tokens: int = 4096):
        self.corpus = corpus
        self.max_tokens = max_tokens

    def execute(self, doc_id: str) -> ToolResult:
        if not doc_id:
            return ToolResult(
                success=False,
                output="",
                error="No document ID provided",
            )

        doc = self.corpus.get_document(doc_id)
        if doc is None:
            return ToolResult(
                success=False,
                output="",
                error=f"Document not found: {doc_id}",
            )

        # Truncate if too long
        content = doc.content
        approx_tokens = len(content.split()) * 1.3

        if approx_tokens > self.max_tokens:
            # Truncate to approximate token limit
            words = content.split()
            max_words = int(self.max_tokens / 1.3)
            content = " ".join(words[:max_words]) + "\n\n[TRUNCATED - document continues...]"

        output = f"# {doc.title}\n\n{content}"

        # Include section overview if available
        if doc.sections:
            section_list = "\n".join(
                f"  - [{s.section_id}] {s.title}"
                for s in doc.sections[:10]
            )
            output = f"# {doc.title}\n\nSections:\n{section_list}\n\n---\n\n{content}"

        return ToolResult(
            success=True,
            output=output,
            metadata={
                "doc_id": doc_id,
                "title": doc.title,
                "token_count": doc.token_count,
                "truncated": approx_tokens > self.max_tokens,
                "num_sections": len(doc.sections),
            },
        )

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "doc_id": {
                    "type": "string",
                    "description": "Document ID from search results (e.g., 'abc123')",
                },
            },
            required=["doc_id"],
        )


class ReadSectionTool(Tool):
    """
    Read a specific section of a document.

    Enables hierarchical navigation (A:B:C pattern).
    Agent learns to navigate efficiently.
    """

    name = "read_section"
    description = "Read a specific section of a document. More focused than reading full document."

    def __init__(self, corpus: DocumentCorpus, max_tokens: int = 2048):
        self.corpus = corpus
        self.max_tokens = max_tokens

    def execute(self, section_id: str) -> ToolResult:
        if not section_id:
            return ToolResult(
                success=False,
                output="",
                error="No section ID provided",
            )

        section = self.corpus.get_section(section_id)
        if section is None:
            # Try to parse as doc_id:section_num
            if ":" in section_id:
                doc_id = section_id.split(":")[0]
                doc = self.corpus.get_document(doc_id)
                if doc:
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"Section not found. Available sections: {[s.section_id for s in doc.sections]}",
                    )
            return ToolResult(
                success=False,
                output="",
                error=f"Section not found: {section_id}",
            )

        content = section.content
        approx_tokens = len(content.split()) * 1.3

        if approx_tokens > self.max_tokens:
            words = content.split()
            max_words = int(self.max_tokens / 1.3)
            content = " ".join(words[:max_words]) + "\n\n[TRUNCATED]"

        # Build context about location in hierarchy
        location_info = f"Document: {section.doc_id}"
        if section.parent_id:
            location_info += f" | Parent: {section.parent_id}"
        if section.children_ids:
            location_info += f" | Children: {section.children_ids}"

        output = f"## {section.title}\n\n{location_info}\n\n---\n\n{content}"

        return ToolResult(
            success=True,
            output=output,
            metadata={
                "section_id": section_id,
                "doc_id": section.doc_id,
                "title": section.title,
                "level": section.level,
                "parent_id": section.parent_id,
                "children_ids": section.children_ids,
                "token_count": section.token_count,
            },
        )

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "section_id": {
                    "type": "string",
                    "description": "Section ID in format 'doc_id:section_num' (e.g., 'abc123:2')",
                },
            },
            required=["section_id"],
        )


class AnswerTool(Tool):
    """
    Submit final answer.

    Terminal action - ends the episode.
    Citation is required to enforce grounding.
    """

    name = "answer"
    description = "Submit your final answer to the question. Must cite sources."

    def execute(self, answer: str, sources: list[str] | None = None) -> ToolResult:
        if not answer or not answer.strip():
            return ToolResult(
                success=False,
                output="",
                error="Empty answer provided",
            )

        output = f"Answer: {answer}"
        if sources:
            output += f"\n\nSources: {', '.join(sources)}"

        return ToolResult(
            success=True,
            output=output,
            metadata={
                "answer": answer,
                "sources": sources or [],
                "is_terminal": True,
            },
        )

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "answer": {
                    "type": "string",
                    "description": "Your final answer to the question",
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of document IDs that support your answer",
                },
            },
            required=["answer"],
        )


class DontKnowTool(Tool):
    """
    Admit uncertainty.

    Key insight: this is a strategically valuable action.
    Better reward than wrong answer, creating epistemic calibration.
    """

    name = "dont_know"
    description = "Use when you cannot find enough evidence to answer confidently."

    def execute(self, reason: str | None = None) -> ToolResult:
        output = "I don't have enough information to answer this question confidently."
        if reason:
            output += f"\n\nReason: {reason}"

        return ToolResult(
            success=True,
            output=output,
            metadata={
                "reason": reason,
                "is_terminal": True,
                "is_dont_know": True,
            },
        )

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "reason": {
                    "type": "string",
                    "description": "Optional: explain why you can't answer",
                },
            },
            required=[],
        )


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Get tool by name."""
        return self.tools.get(name)

    def execute(self, name: str, **kwargs) -> ToolResult:
        """Execute a tool by name."""
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown tool: {name}. Available: {list(self.tools.keys())}",
            )

        valid, error = tool.validate_args(kwargs)
        if not valid:
            return ToolResult(
                success=False,
                output="",
                error=error,
            )

        return tool.execute(**kwargs)

    def get_all_schemas(self) -> list[dict]:
        """Get schemas for all tools."""
        return [tool.get_schema().to_dict() for tool in self.tools.values()]

    def get_tool_descriptions(self) -> str:
        """Get formatted tool descriptions for prompt."""
        lines = ["Available tools:\n"]
        for tool in self.tools.values():
            schema = tool.get_schema()
            params = ", ".join(
                f"{p}: {schema.parameters[p].get('type', 'any')}"
                for p in schema.required
            )
            optional = [p for p in schema.parameters if p not in schema.required]
            if optional:
                params += f" (optional: {', '.join(optional)})"
            lines.append(f"- {tool.name}({params}): {tool.description}")
        return "\n".join(lines)

    @property
    def tool_names(self) -> list[str]:
        """List of available tool names."""
        return list(self.tools.keys())


def create_default_registry(corpus: DocumentCorpus) -> ToolRegistry:
    """Create registry with default research tools."""
    registry = ToolRegistry()
    registry.register(KeywordSearchTool(corpus))
    registry.register(SemanticSearchTool(corpus))
    registry.register(ReadDocumentTool(corpus))
    registry.register(ReadSectionTool(corpus))
    registry.register(AnswerTool())
    registry.register(DontKnowTool())
    return registry
