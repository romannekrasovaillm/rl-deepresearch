"""
Tool-augmented research environment.

Provides a structured environment for research agents with:
- Dual search (keyword + semantic)
- Hierarchical document reading
- State compression (MEM1 insight)
- Structured action space
"""

from .base import ResearchEnvironment, EnvironmentState
from .tools import (
    Tool,
    KeywordSearchTool,
    SemanticSearchTool,
    ReadDocumentTool,
    ReadSectionTool,
    AnswerTool,
    DontKnowTool,
)
from .corpus import DocumentCorpus, Document, Section

__all__ = [
    "ResearchEnvironment",
    "EnvironmentState",
    "Tool",
    "KeywordSearchTool",
    "SemanticSearchTool",
    "ReadDocumentTool",
    "ReadSectionTool",
    "AnswerTool",
    "DontKnowTool",
    "DocumentCorpus",
    "Document",
    "Section",
]
