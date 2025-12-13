"""
Document corpus for research environment.

Implements hierarchical document structure with IDs (A:B:C pattern)
for navigation learning.
"""

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass
class Section:
    """A section within a document."""

    section_id: str  # Format: doc_id:section_num
    title: str
    content: str
    level: int = 1  # Heading level (1, 2, 3, etc.)
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        """Extract document ID from section ID."""
        return self.section_id.split(":")[0]

    @property
    def token_count(self) -> int:
        """Approximate token count."""
        return len(self.content.split()) * 1.3  # Rough estimate

    def to_dict(self) -> dict:
        return {
            "section_id": self.section_id,
            "title": self.title,
            "content": self.content,
            "level": self.level,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "metadata": self.metadata,
        }


@dataclass
class Document:
    """A document in the corpus."""

    doc_id: str
    title: str
    content: str
    sections: list[Section] = field(default_factory=list)
    source: str = ""
    url: str | None = None
    metadata: dict = field(default_factory=dict)
    embedding: np.ndarray | None = None

    def __post_init__(self):
        # Generate sections if not provided
        if not self.sections and self.content:
            self._generate_sections()

    def _generate_sections(self):
        """Split content into sections based on structure."""
        import re

        # Try to detect markdown-style headers
        header_pattern = r'^(#{1,6})\s+(.+)$'
        lines = self.content.split('\n')

        current_section_content = []
        current_title = self.title
        current_level = 1
        section_num = 0

        for line in lines:
            match = re.match(header_pattern, line)
            if match:
                # Save previous section
                if current_section_content:
                    content = '\n'.join(current_section_content).strip()
                    if content:
                        section = Section(
                            section_id=f"{self.doc_id}:{section_num}",
                            title=current_title,
                            content=content,
                            level=current_level,
                        )
                        self.sections.append(section)
                        section_num += 1
                        current_section_content = []

                # Start new section
                current_level = len(match.group(1))
                current_title = match.group(2).strip()
            else:
                current_section_content.append(line)

        # Save last section
        if current_section_content:
            content = '\n'.join(current_section_content).strip()
            if content:
                section = Section(
                    section_id=f"{self.doc_id}:{section_num}",
                    title=current_title,
                    content=content,
                    level=current_level,
                )
                self.sections.append(section)

        # If no sections found, create one from entire content
        if not self.sections:
            self.sections.append(Section(
                section_id=f"{self.doc_id}:0",
                title=self.title,
                content=self.content,
                level=1,
            ))

        # Set up parent-child relationships
        self._build_section_hierarchy()

    def _build_section_hierarchy(self):
        """Build hierarchical relationships between sections."""
        stack: list[Section] = []

        for section in self.sections:
            # Pop sections from stack until we find parent
            while stack and stack[-1].level >= section.level:
                stack.pop()

            if stack:
                section.parent_id = stack[-1].section_id
                stack[-1].children_ids.append(section.section_id)

            stack.append(section)

    def get_section(self, section_id: str) -> Section | None:
        """Get section by ID."""
        for section in self.sections:
            if section.section_id == section_id:
                return section
        return None

    def get_summary(self, max_chars: int = 500) -> str:
        """Get document summary."""
        if len(self.content) <= max_chars:
            return self.content
        return self.content[:max_chars] + "..."

    @property
    def token_count(self) -> int:
        """Approximate token count."""
        return int(len(self.content.split()) * 1.3)

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "content": self.content,
            "sections": [s.to_dict() for s in self.sections],
            "source": self.source,
            "url": self.url,
            "metadata": self.metadata,
        }


class DocumentCorpus:
    """
    Corpus of documents for research environment.

    Supports:
    - BM25 keyword search
    - Semantic search (with embeddings)
    - Hierarchical document IDs for navigation
    """

    def __init__(
        self,
        embedding_model: str | None = None,
        use_bm25: bool = True,
    ):
        self.documents: dict[str, Document] = {}
        self.sections: dict[str, Section] = {}
        self.embedding_model_name = embedding_model
        self.embedder = None
        self.use_bm25 = use_bm25
        self.bm25 = None
        self.doc_index: list[str] = []  # Map BM25 index to doc_id
        self._embeddings_computed = False

    def add_document(self, doc: Document) -> None:
        """Add a document to the corpus."""
        self.documents[doc.doc_id] = doc
        for section in doc.sections:
            self.sections[section.section_id] = section

        # Invalidate indices
        self.bm25 = None
        self._embeddings_computed = False

    def add_documents(self, docs: list[Document]) -> None:
        """Add multiple documents."""
        for doc in docs:
            self.add_document(doc)

    def get_document(self, doc_id: str) -> Document | None:
        """Get document by ID."""
        return self.documents.get(doc_id)

    def get_section(self, section_id: str) -> Section | None:
        """Get section by ID."""
        return self.sections.get(section_id)

    def _build_bm25_index(self) -> None:
        """Build BM25 index for keyword search."""
        from rank_bm25 import BM25Okapi

        self.doc_index = list(self.documents.keys())
        tokenized_corpus = []

        for doc_id in self.doc_index:
            doc = self.documents[doc_id]
            # Tokenize: title + content
            text = f"{doc.title} {doc.content}".lower()
            tokens = text.split()
            tokenized_corpus.append(tokens)

        self.bm25 = BM25Okapi(tokenized_corpus)

    def _compute_embeddings(self) -> None:
        """Compute embeddings for all documents."""
        if self._embeddings_computed or not self.embedding_model_name:
            return

        try:
            from sentence_transformers import SentenceTransformer

            if self.embedder is None:
                self.embedder = SentenceTransformer(self.embedding_model_name)

            texts = []
            doc_ids = []
            for doc_id, doc in self.documents.items():
                texts.append(f"{doc.title}\n{doc.get_summary(1000)}")
                doc_ids.append(doc_id)

            embeddings = self.embedder.encode(texts, show_progress_bar=True)

            for doc_id, embedding in zip(doc_ids, embeddings):
                self.documents[doc_id].embedding = embedding

            self._embeddings_computed = True
        except ImportError:
            pass

    def keyword_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[Document, float]]:
        """
        BM25 keyword search.

        Key insight: keyword search for exact term matching,
        semantic search for conceptual similarity.
        Agent learns when to use each through RL.
        """
        if self.bm25 is None:
            self._build_bm25_index()

        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc_id = self.doc_index[idx]
                doc = self.documents[doc_id]
                results.append((doc, float(scores[idx])))

        return results

    def semantic_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[Document, float]]:
        """
        Semantic search using embeddings.

        Uses cosine similarity between query and document embeddings.
        """
        if not self.embedding_model_name:
            # Fallback to keyword search
            return self.keyword_search(query, top_k)

        self._compute_embeddings()

        if self.embedder is None:
            return self.keyword_search(query, top_k)

        # Encode query
        query_embedding = self.embedder.encode(query)

        # Compute similarities
        similarities = []
        for doc_id, doc in self.documents.items():
            if doc.embedding is not None:
                sim = np.dot(query_embedding, doc.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc.embedding)
                )
                similarities.append((doc, float(sim)))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        keyword_weight: float = 0.5,
    ) -> list[tuple[Document, float]]:
        """
        Hybrid search combining keyword and semantic.

        Useful as a baseline, but RL agents learn to choose
        the right method dynamically.
        """
        keyword_results = self.keyword_search(query, top_k * 2)
        semantic_results = self.semantic_search(query, top_k * 2)

        # Normalize scores
        def normalize(results: list[tuple[Document, float]]) -> dict[str, float]:
            if not results:
                return {}
            max_score = max(r[1] for r in results)
            if max_score == 0:
                return {r[0].doc_id: 0.0 for r in results}
            return {r[0].doc_id: r[1] / max_score for r in results}

        keyword_scores = normalize(keyword_results)
        semantic_scores = normalize(semantic_results)

        # Combine scores
        all_doc_ids = set(keyword_scores.keys()) | set(semantic_scores.keys())
        combined = {}

        for doc_id in all_doc_ids:
            kw_score = keyword_scores.get(doc_id, 0.0)
            sem_score = semantic_scores.get(doc_id, 0.0)
            combined[doc_id] = keyword_weight * kw_score + (1 - keyword_weight) * sem_score

        # Sort and return
        sorted_docs = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [(self.documents[doc_id], score) for doc_id, score in sorted_docs]

    def __len__(self) -> int:
        return len(self.documents)

    def __iter__(self) -> Iterator[Document]:
        return iter(self.documents.values())

    def save(self, path: str | Path) -> None:
        """Save corpus to JSON file."""
        path = Path(path)
        data = {
            "documents": [doc.to_dict() for doc in self.documents.values()],
            "metadata": {
                "embedding_model": self.embedding_model_name,
            }
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "DocumentCorpus":
        """Load corpus from JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        corpus = cls(
            embedding_model=data.get("metadata", {}).get("embedding_model"),
        )

        for doc_data in data["documents"]:
            sections = [Section(**s) for s in doc_data.pop("sections", [])]
            doc = Document(**doc_data, sections=sections)
            corpus.add_document(doc)

        return corpus

    @classmethod
    def from_hotpotqa(cls, split: str = "train", max_docs: int = 10000) -> "DocumentCorpus":
        """Create corpus from HotpotQA supporting documents."""
        try:
            from datasets import load_dataset

            corpus = cls(embedding_model="sentence-transformers/all-MiniLM-L6-v2")

            dataset = load_dataset("hotpot_qa", "fullwiki", split=split)

            seen_titles = set()
            doc_count = 0

            for item in dataset:
                for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
                    if title in seen_titles:
                        continue

                    seen_titles.add(title)
                    content = " ".join(sentences)

                    doc_id = hashlib.md5(title.encode()).hexdigest()[:12]
                    doc = Document(
                        doc_id=doc_id,
                        title=title,
                        content=content,
                        source="hotpotqa",
                    )
                    corpus.add_document(doc)
                    doc_count += 1

                    if doc_count >= max_docs:
                        break

                if doc_count >= max_docs:
                    break

            return corpus
        except Exception as e:
            raise RuntimeError(f"Failed to load HotpotQA: {e}")
