# SPDX-License-Identifier: Apache-2.0
"""Deterministic OpenAI client for CacheBlend + PyramidKV experiments."""

# Future
from __future__ import annotations

# Standard
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import argparse
import asyncio
import collections
import hashlib
import json
import math
import random
import re
import string
import time

# Third Party
from openai import AsyncOpenAI
from transformers import AutoTokenizer

SYSTEM_PROMPT = (
    "You will be asked a question after reading several passages. "
    "Please directly answer the question based on the given passages. "
    "Do NOT repeat the question. The answer should be within 5 words.\n"
    "Passages:\n"
)
QUERY_PROMPT = (
    "\n\nAnswer the question directly based on the given passages. "
    "Do NOT repeat the question. The answer should be within 5 words. "
    "\nQuestion:"
)
SEPARATOR = " # # "
PASSAGE_SPLIT_RE = re.compile(r"(?:^|\n\n)Passage \d+:\n")


def normalize_question(question: str) -> str:
    if not question.endswith("?"):
        question += "?"
    return question[0].lower() + question[1:]


def build_qa_prompt(
    example: dict[str, Any], query_prompt: str
) -> tuple[list[str], str]:
    documents, question, _, _ = normalize_musique_sample(example)
    question = normalize_question(question)
    return documents, f"{query_prompt}{question}\nAnswer:"


def normalize_musique_sample(
    sample: dict[str, Any],
) -> tuple[list[str], str, list[str], str]:
    """Normalize CacheBlend, official MuSiQue, and LongBench records."""
    if isinstance(sample.get("ctxs"), list):
        documents = [
            f"{context.get('title', '')}{context['text']}" for context in sample["ctxs"]
        ]
        question = sample.get("question")
        answers = sample.get("answers")
        group = sample.get("question_type") or f"contexts_{len(documents)}"
    elif sample.get("input") and sample.get("context"):
        documents = [
            passage.strip()
            for passage in PASSAGE_SPLIT_RE.split(sample["context"])
            if passage.strip()
        ]
        question = sample.get("input")
        answers = sample.get("answers")
        group = f"contexts_{len(documents)}"
    elif isinstance(sample.get("paragraphs"), list):
        paragraphs = sorted(
            sample["paragraphs"], key=lambda paragraph: paragraph.get("idx", 0)
        )
        documents = [
            f"{paragraph.get('title', '')}{paragraph.get('paragraph_text', '')}"
            for paragraph in paragraphs
        ]
        question = sample.get("question")
        answer = sample.get("answer")
        aliases = sample.get("answer_aliases") or []
        answers = [answer, *aliases] if answer else aliases
        sample_id = str(sample.get("id", ""))
        group = next(
            (part for part in re.split(r"[_-]", sample_id) if "hop" in part),
            f"contexts_{len(documents)}",
        )
    else:
        raise ValueError("unsupported MuSiQue record format")

    if not question or not documents or not answers:
        raise ValueError("MuSiQue record is missing question, documents, or answers")
    if not all(
        isinstance(document, str) and document.strip() for document in documents
    ):
        raise ValueError("MuSiQue record contains an empty document")
    if not all(isinstance(answer, str) and answer.strip() for answer in answers):
        raise ValueError("MuSiQue record contains an empty answer")
    return documents, str(question), list(dict.fromkeys(answers)), str(group)


def parse_generation(value: str) -> str:
    if not value or not value.strip():
        return ""
    value = value.lstrip("\n").split("\n")[0].strip()
    if value.lower().startswith("yes"):
        return "Yes"
    words = value.split()
    return "No" if words and words[0].lower().startswith("no") else value


def normalize_answer(value: str) -> str:
    value = value.lower()
    value = "".join(
        character for character in value if character not in string.punctuation
    )
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def compute_f1(prediction: str, answer: str, tokenizer: Any) -> float:
    predicted_tokens = tokenizer.encode(
        normalize_answer(parse_generation(prediction)), add_special_tokens=False
    )
    answer_tokens = tokenizer.encode(normalize_answer(answer), add_special_tokens=False)
    common = collections.Counter(predicted_tokens) & collections.Counter(answer_tokens)
    matching = sum(common.values())
    if not predicted_tokens or not answer_tokens:
        return float(predicted_tokens == answer_tokens)
    if not matching:
        return 0.0
    precision = matching / len(predicted_tokens)
    recall = matching / len(answer_tokens)
    return 2 * precision * recall / (precision + recall)


@dataclass
class WorkItem:
    request_id: str
    prompt: str | list[int]
    prompt_tokens: int
    answers: list[str]
    sample_index: int | None = None
    quality_group: str | None = None


@dataclass
class RequestResult:
    request_id: str
    prompt_tokens: int
    output_tokens: int
    ttft_s: float | None
    tpot_s: float | None
    itl_s: list[float]
    e2e_s: float
    generated_text: str
    generated_token_ids: list[int]
    token_ids_from_server: bool
    answers: list[str]
    sample_index: int | None
    quality_group: str | None
    error: str | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def token_stream(tokenizer: Any, seed: int, length: int) -> list[int]:
    if length < 0:
        raise ValueError(f"negative token stream length: {length}")
    vocabulary = tokenizer.encode(
        "CacheBlend PyramidKV deterministic document context evidence retrieval ",
        add_special_tokens=False,
    )
    if not vocabulary:
        raise RuntimeError("tokenizer produced no ordinary tokens")
    rng = random.Random(seed)
    return [vocabulary[rng.randrange(len(vocabulary))] for _ in range(length)]


def fixed_items(
    tokenizer: Any,
    input_length: int,
    count: int,
    seed: int,
) -> tuple[list[WorkItem], list[list[int]]]:
    system = tokenizer.encode(SYSTEM_PROMPT, add_special_tokens=True)
    separator = tokenizer.encode(SEPARATOR, add_special_tokens=False)
    query_length = min(128, max(32, input_length // 8))
    document_count = 8 if input_length >= 4096 else 4
    overhead = len(system) + len(separator) * (document_count + 1) + query_length
    document_budget = input_length - overhead
    if document_budget < document_count * 32:
        raise ValueError(
            f"input length {input_length} is too short for fixed RAG prompt"
        )

    base_size, extra = divmod(document_budget, document_count)
    documents = [
        token_stream(
            tokenizer,
            seed + 10_000 + index,
            base_size + (index < extra),
        )
        for index in range(document_count)
    ]
    precompute = [system + separator + document + separator for document in documents]
    if any(len(prompt) >= 4096 for prompt in precompute):
        raise ValueError("a document precompute prompt crossed the PyramidKV threshold")

    items = []
    for request_index in range(count):
        order = list(range(document_count))
        random.Random(seed + request_index).shuffle(order)
        prompt = list(system)
        for document_index in order:
            prompt.extend(separator)
            prompt.extend(documents[document_index])
        prompt.extend(separator)
        prompt.extend(
            token_stream(tokenizer, seed + 100_000 + request_index, query_length)
        )
        if len(prompt) != input_length:
            raise AssertionError(f"built {len(prompt)} tokens, expected {input_length}")
        items.append(
            WorkItem(
                request_id=f"fixed-{input_length}-{request_index:04d}",
                prompt=prompt,
                prompt_tokens=len(prompt),
                answers=[],
            )
        )
    return items, precompute


def mixed_items(
    tokenizer: Any,
    short_input_length: int,
    long_input_length: int,
    count: int,
    seed: int,
) -> tuple[list[WorkItem], list[list[int]]]:
    short_count = (count + 1) // 2
    long_count = count // 2
    short_items, short_precompute = fixed_items(
        tokenizer, short_input_length, short_count, seed
    )
    long_items, long_precompute = fixed_items(
        tokenizer, long_input_length, long_count, seed + 1_000_000
    )
    items: list[WorkItem] = []
    for index in range(max(short_count, long_count)):
        if index < short_count:
            short_items[index].request_id = f"mixed-short-{index:04d}"
            items.append(short_items[index])
        if index < long_count:
            long_items[index].request_id = f"mixed-long-{index:04d}"
            items.append(long_items[index])
    return items, [*short_precompute, *long_precompute]


def load_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"dataset is empty: {path}")
    if text.startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(records, list) or not all(
        isinstance(record, dict) for record in records
    ):
        raise ValueError("dataset must contain a JSON list or JSONL objects")
    return records


def _valid_musique_sample(sample: Any) -> bool:
    if not isinstance(sample, dict):
        return False
    try:
        normalize_musique_sample(sample)
    except (TypeError, ValueError):
        return False
    return True


def rag_items(
    tokenizer: Any,
    dataset_path: Path,
    sample_list_path: Path,
    count: int,
    min_prompt_tokens: int = 0,
) -> tuple[list[WorkItem], list[list[int]], str]:
    dataset = load_json_records(dataset_path)

    def build_sample(
        sample: dict[str, Any],
    ) -> tuple[list[list[int]], list[int], int, str]:
        documents, question, _, quality_group = normalize_musique_sample(sample)
        question_prompt = f"{QUERY_PROMPT}{normalize_question(question)}\nAnswer:"
        system_tokens = tokenizer.encode(SYSTEM_PROMPT, add_special_tokens=True)
        separator_tokens = tokenizer.encode(SEPARATOR, add_special_tokens=False)
        document_tokens = [
            tokenizer.encode(document, add_special_tokens=False)
            for document in documents
        ]
        prompt = list(system_tokens)
        for token_ids in document_tokens:
            prompt.extend(separator_tokens)
            prompt.extend(token_ids)
        prompt.extend(separator_tokens)
        prompt.extend(tokenizer.encode(question_prompt, add_special_tokens=False))
        prompt_tokens = len(prompt)
        precompute_prompts = [
            [*system_tokens, *separator_tokens, *token_ids, *separator_tokens]
            for token_ids in document_tokens
        ]
        if prompt_tokens > 8192:
            raise ValueError("MuSiQue prompt exceeds max_model_len=8192")
        if any(len(value) >= 4096 for value in precompute_prompts):
            raise ValueError("MuSiQue document precompute crossed PyramidKV threshold")
        return precompute_prompts, prompt, prompt_tokens, quality_group

    if sample_list_path.exists():
        sample_indices = json.loads(sample_list_path.read_text(encoding="utf-8"))
    else:
        sample_indices = []
        for index, sample in enumerate(dataset):
            if not _valid_musique_sample(sample):
                continue
            try:
                _, _, prompt_tokens, _ = build_sample(sample)
            except ValueError:
                continue
            if prompt_tokens < min_prompt_tokens:
                continue
            sample_indices.append(index)
            if len(sample_indices) == count:
                break
        if len(sample_indices) != count:
            raise ValueError(
                f"dataset has only {len(sample_indices)} valid samples, "
                f"expected {count}"
            )
        write_json(sample_list_path, sample_indices)
    if len(sample_indices) != count or len(set(sample_indices)) != count:
        raise ValueError(
            "sample list must contain the requested number of unique indices"
        )

    items: list[WorkItem] = []
    documents: dict[tuple[int, ...], list[int]] = {}
    for request_index, sample_index in enumerate(sample_indices):
        sample = dataset[sample_index]
        if not _valid_musique_sample(sample):
            raise ValueError(
                f"sample index {sample_index} is not a valid MuSiQue sample"
            )
        sample_documents, prompt, prompt_tokens, quality_group = build_sample(sample)
        if prompt_tokens < min_prompt_tokens:
            raise ValueError(
                f"sample index {sample_index} has {prompt_tokens} prompt tokens, "
                f"below the required minimum {min_prompt_tokens}"
            )
        for document in sample_documents:
            documents.setdefault(tuple(document), document)
        _, _, answers, _ = normalize_musique_sample(sample)
        items.append(
            WorkItem(
                request_id=f"musique-{request_index:04d}",
                prompt=prompt,
                prompt_tokens=prompt_tokens,
                answers=answers,
                sample_index=sample_index,
                quality_group=quality_group,
            )
        )
    return items, list(documents.values()), sha256_file(sample_list_path)


async def precompute_documents(
    client: AsyncOpenAI,
    model: str,
    prompts: list[str | list[int]],
    settle_seconds: float,
) -> tuple[int, float]:
    started = time.perf_counter()
    completed = 0
    for index, prompt in enumerate(prompts):
        await client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=1,
            temperature=0,
            extra_body={"ignore_eos": True},
            extra_headers={"x-request-id": f"precompute-{index:05d}"},
        )
        completed += 1
    if settle_seconds:
        await asyncio.sleep(settle_seconds)
    return completed, time.perf_counter() - started


def resolve_precompute_settle_seconds(
    document_count: int, configured_seconds: float | None
) -> float:
    if configured_seconds is not None:
        return configured_seconds
    return max(5.0, document_count * 0.25)


async def execute_item(
    client: AsyncOpenAI,
    tokenizer: Any,
    model: str,
    item: WorkItem,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> RequestResult:
    async with semaphore:
        started = time.perf_counter()
        arrivals: list[float] = []
        pieces: list[str] = []
        streamed_token_ids: list[int] = []
        server_returned_token_ids = False
        error = None
        try:
            stream = await client.completions.create(
                model=model,
                prompt=item.prompt,
                max_tokens=max_tokens,
                temperature=0,
                stream=True,
                extra_body={"ignore_eos": True, "return_token_ids": True},
                extra_headers={"x-request-id": item.request_id},
            )
            async for chunk in stream:
                choice = chunk.choices[0]
                piece = choice.text
                token_ids = getattr(choice, "token_ids", None)
                if token_ids is None:
                    token_ids = (getattr(choice, "model_extra", None) or {}).get(
                        "token_ids"
                    )
                if token_ids is not None:
                    server_returned_token_ids = True
                    streamed_token_ids.extend(token_ids)
                if piece:
                    pieces.append(piece)
                if piece or token_ids:
                    arrivals.append(time.perf_counter())
        except Exception as exc:  # The raw result must retain request failures.
            error = f"{type(exc).__name__}: {exc}"
        finished = time.perf_counter()

    generated_text = "".join(pieces)
    generated_ids = (
        streamed_token_ids
        if server_returned_token_ids
        else tokenizer.encode(generated_text, add_special_tokens=False)
    )
    if error is None and not arrivals:
        error = "empty completion stream"
    ttft = arrivals[0] - started if arrivals else None
    itls = [
        later - earlier for earlier, later in zip(arrivals, arrivals[1:], strict=False)
    ]
    e2e = finished - started
    tpot = (
        (e2e - ttft) / (len(generated_ids) - 1)
        if len(generated_ids) > 1 and ttft is not None and math.isfinite(ttft)
        else None
    )
    return RequestResult(
        request_id=item.request_id,
        prompt_tokens=item.prompt_tokens,
        output_tokens=len(generated_ids),
        ttft_s=ttft,
        tpot_s=tpot,
        itl_s=itls,
        e2e_s=e2e,
        generated_text=generated_text,
        generated_token_ids=generated_ids,
        token_ids_from_server=server_returned_token_ids,
        answers=item.answers,
        sample_index=item.sample_index,
        quality_group=item.quality_group,
        error=error,
    )


async def run(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    dataset_sha = None
    sample_list_sha = None
    if args.workload == "fixed":
        items, precompute = fixed_items(
            tokenizer, args.input_length, args.requests, args.seed
        )
    elif args.workload == "mixed":
        items, precompute = mixed_items(
            tokenizer,
            args.short_input_length,
            args.long_input_length,
            args.requests,
            args.seed,
        )
    else:
        dataset_path = Path(args.dataset).resolve()
        items, precompute, sample_list_sha = rag_items(
            tokenizer,
            dataset_path,
            Path(args.sample_list).resolve(),
            args.requests,
            args.min_prompt_tokens,
        )
        dataset_sha = sha256_file(dataset_path)

    if args.cache_state == "cold":
        precompute = []
    elif args.cache_state == "partial":
        precompute = precompute[: max(1, len(precompute) // 2)]

    client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.timeout,
    )
    precomputed = 0
    precompute_s = 0.0
    precompute_settle_s = 0.0
    if args.mode in {"C", "CP"} and precompute:
        precompute_settle_s = resolve_precompute_settle_seconds(
            len(precompute), args.precompute_settle_seconds
        )
        precomputed, precompute_s = await precompute_documents(
            client,
            args.model,
            precompute,
            precompute_settle_s,
        )

    benchmark_started = time.perf_counter()
    semaphore = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(
        *(
            execute_item(
                client,
                tokenizer,
                args.model,
                item,
                args.max_tokens,
                semaphore,
            )
            for item in items
        )
    )
    duration = time.perf_counter() - benchmark_started
    await client.close()

    successes = [result for result in results if result.error is None]
    total_output = sum(result.output_tokens for result in successes)
    quality_f1 = None
    quality_em = None
    f1_by_group = None
    em_by_group = None
    if args.workload == "rag" and successes:
        f1_values = {
            result.request_id: max(
                compute_f1(result.generated_text, answer, tokenizer)
                for answer in result.answers
            )
            for result in successes
        }
        em_values = {
            result.request_id: max(
                normalize_answer(parse_generation(result.generated_text))
                == normalize_answer(answer)
                for answer in result.answers
            )
            for result in successes
        }
        quality_f1 = 100 * sum(f1_values.values()) / len(f1_values)
        quality_em = 100 * sum(em_values.values()) / len(em_values)
        groups = sorted({result.quality_group for result in successes})
        f1_by_group = {
            group: 100
            * sum(
                f1_values[result.request_id]
                for result in successes
                if result.quality_group == group
            )
            / sum(result.quality_group == group for result in successes)
            for group in groups
        }
        em_by_group = {
            group: 100
            * sum(
                em_values[result.request_id]
                for result in successes
                if result.quality_group == group
            )
            / sum(result.quality_group == group for result in successes)
            for group in groups
        }

    payload = {
        "schema_version": 1,
        "benchmark_type": args.workload,
        "metadata": {
            "mode": args.mode,
            "repeat": args.repeat,
            "seed": args.seed,
            "concurrency": args.concurrency,
            "cache_state": args.cache_state,
            "input_length": args.input_length if args.workload == "fixed" else None,
            "input_lengths": (
                [args.short_input_length, args.long_input_length]
                if args.workload == "mixed"
                else None
            ),
            "output_length": args.max_tokens,
            "requests": args.requests,
            "model": args.model,
            "dataset_sha256": dataset_sha,
            "sample_list_sha256": sample_list_sha,
            "scheduler_mode": args.scheduler_mode,
            "min_prompt_tokens": args.min_prompt_tokens,
        },
        "precompute": {
            "documents": precomputed,
            "duration_s": precompute_s,
            "settle_s": precompute_settle_s,
        },
        "summary": {
            "duration_s": duration,
            "completed": len(successes),
            "failed": len(results) - len(successes),
            "request_throughput": len(successes) / duration,
            "output_throughput": total_output / duration,
            "f1": quality_f1,
            "em": quality_em,
            "f1_by_group": f1_by_group,
            "em_by_group": em_by_group,
        },
        "requests": [asdict(result) for result in results],
    }
    write_json(Path(args.output), payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("workload", choices=("fixed", "mixed", "rag"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="dummy-key")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--mode", choices=("B", "C", "P", "CP"), required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument(
        "--cache-state", choices=("cold", "partial", "full"), default="full"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-length", type=int)
    parser.add_argument("--short-input-length", type=int)
    parser.add_argument("--long-input-length", type=int)
    parser.add_argument("--scheduler-mode", choices=("async", "sync"), default="async")
    parser.add_argument("--dataset")
    parser.add_argument("--sample-list")
    parser.add_argument("--min-prompt-tokens", type=int, default=0)
    parser.add_argument("--precompute-settle-seconds", type=float)
    args = parser.parse_args()
    if args.workload == "fixed" and args.input_length is None:
        parser.error("fixed workload requires --input-length")
    if args.workload == "mixed" and (
        args.short_input_length is None or args.long_input_length is None
    ):
        parser.error(
            "mixed workload requires --short-input-length and --long-input-length"
        )
    if args.workload == "rag" and (not args.dataset or not args.sample_list):
        parser.error("rag workload requires --dataset and --sample-list")
    if args.min_prompt_tokens < 0:
        parser.error("--min-prompt-tokens must be non-negative")
    if (
        args.precompute_settle_seconds is not None
        and args.precompute_settle_seconds < 0
    ):
        parser.error("--precompute-settle-seconds must be non-negative")
    return args


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
