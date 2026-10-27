#!/usr/bin/env python3
"""Record/audit pruning development evaluations, adapted from the audited Day 7 recorder.

The forwarding recorder accepts only the frozen development prompts. It records
raw responses, checks prompt token counts, and cross-checks ARC likelihoods
against the harness samples. It never stops the user's vLLM server.
"""

import argparse
import copy
from collections import defaultdict, deque
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import distribution, version
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from transformers import AutoTokenizer

TASKS = ("day7_gsm8k_dev", "day7_arc_challenge_dev")
BASE = "http://127.0.0.1:8000"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def http_json(path):
    with build_opener(ProxyHandler({})).open(BASE + path, timeout=10) as response:
        return response.status, json.load(response)


def one_prompt(payload):
    prompt = payload["prompt"]
    if isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], list):
        prompt = prompt[0]
    require(isinstance(prompt, list) and prompt and
            all(type(token) is int for token in prompt),
            "Expected one tokenized prompt per API request")
    return prompt


class Recorder(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, directory, expected, alias):
        super().__init__(("127.0.0.1", 0), Handler)
        self.directory, self.expected, self.alias = directory, expected, alias
        self.lock = threading.Lock()
        self.sequence = 0


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        server = self.server
        with server.lock:
            server.sequence += 1
            number = server.sequence
        receipt = {"sequence": number, "path": self.path}
        start = time.perf_counter()
        status, result = 502, {"error": "Forwarding did not complete"}
        try:
            require(self.path == "/v1/completions", "Unexpected endpoint")
            size = int(self.headers.get("Content-Length", "0"))
            require(0 < size <= 10 * 1024 * 1024, "Invalid request body size")
            body = self.rfile.read(size)
            payload = json.loads(body)
            receipt["request"] = payload
            require(payload["model"] == server.alias, "Unexpected model alias")
            require(not payload.get("stream", False), "Recorder expects non-streaming requests")
            require(payload.get("temperature") == 0, "Expected greedy requests")
            require(payload.get("seed") == 42, "Request seed differs from 42")
            prompt = one_prompt(payload)
            kind = "arc" if payload.get("echo", False) else "math"
            if kind == "math":
                require(payload["max_tokens"] == 2048 and payload.get("top_p", 1) == 1,
                        "Math generation settings differ")
                require(set(("Question:", "</s>", "<|im_end|>")) <= set(payload.get("stop", [])),
                        "Expected math stop strings are missing")
            else:
                require(payload["max_tokens"] == 1 and payload.get("logprobs") == 1,
                        "Expected echoed ARC token likelihoods")
            with server.lock:
                matches = server.expected.get((kind, tuple(prompt)))
                require(matches is not None and len(matches) > 0,
                        "Prompt differs from the frozen pilot, or was sent more than once")
                receipt["expected"] = matches.popleft()
            request = Request(BASE + self.path, data=body,
                              headers={"Content-Type": "application/json"})
            opener = build_opener(ProxyHandler({}))
            try:
                with opener.open(request, timeout=300) as response:
                    status, raw = response.status, response.read()
            except HTTPError as error:
                status, raw = error.code, error.read()
            result = json.loads(raw)
            receipt.update(http_status=status, response=result)
        except Exception as error:
            receipt["error"] = f"{type(error).__name__}: {error}"
            status, result = 502, {"error": receipt["error"]}
        finally:
            receipt["wall_seconds"] = time.perf_counter() - start
            save_json(server.directory / f"record_{number:06d}.json", receipt)
        output = json.dumps(result, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(output)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(output)
        self.close_connection = True


def loglikelihood(value):
    """Read a saved likelihood pair, including string-serialized numbers."""
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    require(isinstance(value, list) and len(value) == 2, "Unexpected likelihood pair")
    score, flag = value
    require(not isinstance(score, bool), "Invalid likelihood score")
    try:
        score = float(score)
    except (TypeError, ValueError):
        raise RuntimeError("Likelihood score is not numeric")
    require(math.isfinite(score), "Likelihood score is not finite")
    require(type(flag) is bool or
            (isinstance(flag, str) and flag in ("True", "False")),
            "Invalid greedy flag")
    return score


def audit_saved(out, alias, counts, expected, total, original_wall_seconds=None):
    """Audit saved requests/results without contacting any model server."""
    raw_dir = out / "http_records"
    models = read_json(out / "models.json")
    cards = [card for card in models["data"] if card["id"] == alias]
    require(len(cards) == 1 and cards[0].get("max_model_len") == 8192,
            "Saved server identity/context differs")
    receipts = [read_json(path) for path in sorted(raw_dir.glob("record_*.json"))]
    require(len(receipts) == total, "Unexpected API request count")
    math_receipts, arc_receipts = {}, {}
    for item in receipts:
        require("error" not in item and item.get("http_status") == 200,
                f"Failed recorded request {item['sequence']}")
        expected_item, response = item["expected"], item["response"]
        payload = item["request"]
        kind = "arc" if payload.get("echo", False) else "math"
        matches = expected.get((kind, tuple(one_prompt(payload))))
        require(matches is not None and len(matches) > 0,
                "Saved prompt differs from the frozen pilot or is duplicated")
        require(matches.popleft() == expected_item,
                "Saved prompt/document identity differs from the prepared request")
        require(payload["model"] == alias and payload.get("seed") == 42
                and payload.get("temperature") == 0,
                "Saved generation settings differ")
        require(payload["max_tokens"] == (2048 if kind == "math" else 1),
                "Saved token limit differs")
        if kind == "math":
            require(payload.get("top_p", 1) == 1 and
                    set(("Question:", "</s>", "<|im_end|>")) <= set(payload.get("stop", [])),
                    "Saved math sampling/stop settings differ")
        else:
            require(payload.get("echo") is True and payload.get("logprobs") == 1,
                    "Saved ARC echo/logprob settings differ")
        require(response["model"] == alias and len(response["choices"]) == 1,
                "Unexpected response model or choice count")
        require(response["usage"]["prompt_tokens"] == expected_item["prompt_tokens"],
                "Server prompt token count differs from the frozen request")
        choice = response["choices"][0]
        if expected_item["kind"] == "math":
            require(choice["finish_reason"] in ("stop", "length"), "Unexpected math finish reason")
            require(0 < response["usage"]["completion_tokens"] <= 2048,
                    "Unexpected math output token count")
            math_receipts[expected_item["doc_id"]] = item
        else:
            require(response["usage"]["completion_tokens"] == 1,
                    "Expected one generated token after the echoed ARC prompt")
            values = choice["logprobs"]["token_logprobs"]
            require(len(values) == expected_item["prompt_tokens"] + 1,
                    "Expected all prompt logprobs followed by one generated token")
            terms = values[expected_item["context_tokens"]:expected_item["prompt_tokens"]]
            require(terms and all(isinstance(x, (int, float)) and math.isfinite(x) for x in terms),
                    "ARC continuation logprobs are missing or invalid")
            item["recomputed_loglikelihood"] = sum(terms)
            arc_receipts[(expected_item["doc_id"], expected_item["choice_index"])] = item

    require(sum(len(items) for items in expected.values()) == 0,
            "Saved run is missing frozen requests")
    require(len(math_receipts) == counts[TASKS[0]], "Saved math request count differs")

    results = []
    for path in (out / "harness").rglob("*.json"):
        data = read_json(path)
        if isinstance(data, dict) and set(TASKS) <= set(data.get("results", {})):
            results.append((path, data))
    require(len(results) == 1, "Could not identify a unique harness results file")
    samples_by_task = {}
    sample_filter_counts = {}
    for task in TASKS:
        limit = counts[task]
        paths = [p for p in (out / "harness").rglob("*.jsonl") if task in p.name]
        require(len(paths) == 1, f"Expected one samples file for {task}")
        samples = [json.loads(line) for line in paths[0].read_text().splitlines() if line.strip()]
        expected_filters = ({"strict-match", "flexible-extract"}
                            if task == TASKS[0] else {"none"})
        pairs = [(sample["doc_id"], sample.get("filter")) for sample in samples]
        require(all(type(doc_id) is int for doc_id, _ in pairs),
                "Expected integer harness document IDs")
        require(len(pairs) == limit * len(expected_filters)
                and len(set(pairs)) == len(pairs)
                and set(pairs) == {(doc_id, name) for doc_id in range(limit)
                                  for name in expected_filters},
                f"Missing or duplicate document/filter sample pairs for {task}")
        sample_filter_counts[task] = {
            name: sum(sample["filter"] == name for sample in samples)
            for name in sorted(expected_filters)
        }
        samples_by_task[task] = samples
        for sample in samples:
            doc_id = sample["doc_id"]
            if task == TASKS[0]:
                raw = math_receipts[doc_id]["response"]["choices"][0]["text"]
                generated = sample["resps"][0]
                while isinstance(generated, list) and len(generated) == 1:
                    generated = generated[0]
                require(isinstance(generated, str) and generated.strip() == raw.strip(),
                        "Harness math text differs from the recorded response")
            else:
                for index, value in enumerate(sample["resps"]):
                    raw_ll = arc_receipts[(doc_id, index)]["recomputed_loglikelihood"]
                    require(math.isclose(loglikelihood(value), raw_ll, rel_tol=1e-8, abs_tol=1e-5),
                            "ARC harness loglikelihood differs from the raw token probabilities")

    metrics = results[0][1]["results"]
    length_finishes = sum(x["response"]["choices"][0]["finish_reason"] == "length"
                          for x in math_receipts.values())
    for task, samples in samples_by_task.items():
        for name in sample_filter_counts[task]:
            group = [sample for sample in samples if sample["filter"] == name]
            names = ("exact_match",) if task == TASKS[0] else ("acc", "acc_norm")
            for metric in names:
                values = [sample[metric] for sample in group]
                require(all(isinstance(x, (int, float)) and math.isfinite(x)
                            and 0 <= x <= 1 for x in values), "Invalid per-question metric")
                require(math.isclose(sum(values) / len(values),
                                     metrics[task][f"{metric},{name}"], abs_tol=1e-10),
                        "Per-filter sample scores differ from the aggregate result")
    hashes = {str(p.relative_to(out)): sha256(p) for p in out.rglob("*")
              if p.is_file() and p.name != "quality_summary.json"}
    summary = {"scope": "pruning development evaluation", "questions_per_task": counts,
               "math_length_finishes": length_finishes, "http_requests_verified": total,
               "wall_seconds_including_client_startup": original_wall_seconds,
               "recorded_api_wall_seconds": sum(item["wall_seconds"] for item in receipts),
               "sample_records_per_filter": sample_filter_counts,
               "auditor_sha256": sha256(Path(__file__).resolve()),
               "metrics": metrics, "raw_response_and_result_sha256": hashes}
    save_json(out / "quality_summary.json", summary)
    print(f"\nMath strict-match: {100 * metrics[TASKS[0]]['exact_match,strict-match']:.2f}%")
    print(f"Science acc_norm: {100 * metrics[TASKS[1]]['acc_norm,none']:.2f}%")
    print(f"Math length finishes: {length_finishes}/{counts[TASKS[0]]}")
    print(f"Saved: {out}")
    print("PASS: raw responses, exact prompt tokens and ARC likelihoods match the harness.")
    print("Development results only; previously used data, not fresh unseen test results.")

