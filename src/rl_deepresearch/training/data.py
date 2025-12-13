"""
Dataset and data loading utilities.

Key insights implemented:
- Difficulty filtering (only informative examples)
- Decontamination (filter parametric knowledge)
- Multi-source composition (prevent shortcuts)
"""

import json
import random
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset


class ResearchDataset(Dataset):
    """
    Dataset for research QA training.

    Features:
    - Difficulty filtering
    - Decontamination
    - Multi-hop requirement
    """

    def __init__(
        self,
        data: list[dict],
        min_difficulty: float = 0.0,
        max_difficulty: float = 1.0,
        require_multi_hop: bool = False,
        decontaminate: bool = False,
        decontamination_model: Any = None,
    ):
        self.data = data
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.require_multi_hop = require_multi_hop

        # Filter data
        self.filtered_data = self._filter_data()

        # Decontamination
        if decontaminate and decontamination_model:
            self.filtered_data = self._decontaminate(decontamination_model)

    def _filter_data(self) -> list[dict]:
        """Filter data by difficulty and requirements."""
        filtered = []

        for item in self.data:
            # Difficulty filter
            difficulty = item.get("difficulty", 0.5)
            if not (self.min_difficulty <= difficulty <= self.max_difficulty):
                continue

            # Multi-hop filter
            if self.require_multi_hop:
                is_multi_hop = item.get("multi_hop", False)
                num_sources = len(item.get("sources", []))
                if not is_multi_hop and num_sources < 2:
                    continue

            filtered.append(item)

        return filtered

    def _decontaminate(self, model: Any) -> list[dict]:
        """
        Remove questions model can answer from parametric memory.

        Key insight: without decontamination, can't measure retrieval improvement.
        Model might "cheat" by answering from pretraining.
        """
        decontaminated = []

        for item in self.filtered_data:
            # Try to answer without retrieval
            question = item["question"]
            expected = item.get("answer", "")

            try:
                output = model.generate(
                    prompt=f"Answer briefly: {question}",
                    max_new_tokens=100,
                    temperature=0.0,  # Deterministic
                )

                # Check if model knows answer
                answer = output.text.lower()
                expected_lower = expected.lower()

                # Simple overlap check
                if expected_lower in answer:
                    continue  # Model knows this, skip

                # Word overlap check
                expected_words = set(expected_lower.split())
                answer_words = set(answer.split())
                overlap = len(expected_words & answer_words) / len(expected_words) if expected_words else 0

                if overlap > 0.7:
                    continue  # Model likely knows this

            except Exception:
                pass  # Keep on error

            decontaminated.append(item)

        return decontaminated

    def __len__(self) -> int:
        return len(self.filtered_data)

    def __getitem__(self, idx: int) -> dict:
        return self.filtered_data[idx]

    @classmethod
    def from_hotpotqa(
        cls,
        split: str = "train",
        max_samples: int = 10000,
        **kwargs,
    ) -> "ResearchDataset":
        """Load from HotpotQA dataset."""
        try:
            from datasets import load_dataset

            dataset = load_dataset("hotpot_qa", "fullwiki", split=split)

            data = []
            for i, item in enumerate(dataset):
                if i >= max_samples:
                    break

                # Extract supporting facts as sources
                sources = list(set(item["supporting_facts"]["title"]))

                # Determine difficulty (multi-hop = harder)
                difficulty = 0.7 if item["type"] == "comparison" else 0.5
                if item["level"] == "hard":
                    difficulty += 0.2

                data.append({
                    "question": item["question"],
                    "answer": item["answer"],
                    "sources": sources,
                    "multi_hop": len(sources) > 1,
                    "difficulty": min(difficulty, 1.0),
                    "type": item["type"],
                })

            return cls(data, **kwargs)

        except Exception as e:
            raise RuntimeError(f"Failed to load HotpotQA: {e}")

    @classmethod
    def from_json(cls, path: str | Path, **kwargs) -> "ResearchDataset":
        """Load from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(data, **kwargs)

    def save(self, path: str | Path) -> None:
        """Save to JSON file."""
        with open(path, "w") as f:
            json.dump(self.filtered_data, f, indent=2)


class StreamingResearchDataset(IterableDataset):
    """
    Streaming dataset for large-scale training.

    Useful when data doesn't fit in memory.
    """

    def __init__(
        self,
        data_path: str | Path,
        batch_size: int = 4,
        shuffle_buffer: int = 1000,
    ):
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer

    def __iter__(self) -> Iterator[dict]:
        buffer = []

        with open(self.data_path, "r") as f:
            for line in f:
                item = json.loads(line)
                buffer.append(item)

                if len(buffer) >= self.shuffle_buffer:
                    random.shuffle(buffer)
                    while len(buffer) > self.shuffle_buffer // 2:
                        yield buffer.pop()

        # Flush remaining
        random.shuffle(buffer)
        for item in buffer:
            yield item


class DataCollator:
    """
    Collate function for research data.

    Handles:
    - Grouping for GRPO
    - Tokenization
    - Padding
    """

    def __init__(
        self,
        tokenizer: Any = None,
        max_length: int = 2048,
        group_size: int = 1,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.group_size = group_size

    def __call__(self, batch: list[dict]) -> list[dict]:
        """
        Collate batch for training.

        For GRPO, we don't tokenize here - just return questions.
        Tokenization happens after rollout generation.
        """
        return batch


class CurriculumDataset(Dataset):
    """
    Dataset with curriculum learning support.

    Key insight: curriculum from data distribution,
    not reward complexity. Start easy, increase difficulty.
    """

    def __init__(
        self,
        data: list[dict],
        num_stages: int = 3,
        current_stage: int = 0,
    ):
        self.full_data = data
        self.num_stages = num_stages
        self.current_stage = current_stage

        # Sort by difficulty
        self.sorted_data = sorted(data, key=lambda x: x.get("difficulty", 0.5))

        # Update active data
        self._update_active_data()

    def _update_active_data(self) -> None:
        """Update data for current curriculum stage."""
        stage_size = len(self.sorted_data) // self.num_stages
        end_idx = min((self.current_stage + 1) * stage_size, len(self.sorted_data))

        # Include all data up to current stage
        self.data = self.sorted_data[:end_idx]

    def advance_stage(self) -> bool:
        """
        Advance to next curriculum stage.

        Returns True if advanced, False if already at max.
        """
        if self.current_stage >= self.num_stages - 1:
            return False

        self.current_stage += 1
        self._update_active_data()
        return True

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]


class InformativeSampler:
    """
    Sampler that prioritizes informative examples.

    Key insight (DUPO): not all examples equally useful.
    Examples with zero variance provide no gradient signal.
    """

    def __init__(
        self,
        dataset: Dataset,
        reward_history: dict[str, list[float]] | None = None,
        duplicate_informative: bool = True,
        max_duplicates: int = 3,
    ):
        self.dataset = dataset
        self.reward_history = reward_history or {}
        self.duplicate_informative = duplicate_informative
        self.max_duplicates = max_duplicates

    def get_informative_indices(self) -> list[int]:
        """Get indices of informative examples."""
        indices = list(range(len(self.dataset)))

        if not self.reward_history:
            return indices

        informative = []

        for idx in indices:
            item = self.dataset[idx]
            key = item.get("id", item.get("question", str(idx)))

            if key in self.reward_history:
                rewards = self.reward_history[key]
                # Check variance
                if len(rewards) >= 2:
                    variance = sum((r - sum(rewards)/len(rewards))**2 for r in rewards) / len(rewards)
                    if variance > 0.01:  # Has informative variance
                        # Duplicate informative examples
                        if self.duplicate_informative:
                            num_copies = min(self.max_duplicates, int(variance * 10) + 1)
                            informative.extend([idx] * num_copies)
                        else:
                            informative.append(idx)
                    else:
                        informative.append(idx)  # Keep but don't duplicate
                else:
                    informative.append(idx)  # Not enough history
            else:
                informative.append(idx)  # Never seen

        return informative

    def update_history(self, question_id: str, rewards: list[float]) -> None:
        """Update reward history for a question."""
        if question_id not in self.reward_history:
            self.reward_history[question_id] = []
        self.reward_history[question_id].extend(rewards)

        # Keep only recent history
        if len(self.reward_history[question_id]) > 100:
            self.reward_history[question_id] = self.reward_history[question_id][-100:]


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    collate_fn: DataCollator | None = None,
) -> DataLoader:
    """Create dataloader with research-specific settings."""
    if collate_fn is None:
        collate_fn = DataCollator()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
