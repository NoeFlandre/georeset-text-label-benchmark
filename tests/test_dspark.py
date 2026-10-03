"""Contract tests for direct EUNIS label generation with the pinned DSpark runtime."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow.parquet as pq
import pytest

from georeset_text_label_benchmark.pilot import cli, dspark, dspark_runner, runner
from georeset_text_label_benchmark.pilot.protocol import (
    CANDIDATE_LABELS_SHA256,
    OVERLAP_DATASET,
    OVERLAP_PARQUET_SHA256,
    OVERLAP_REVISION,
)
from georeset_text_label_benchmark.pilot.sampling import select_distinct_sample


def _candidate_file() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "pilot_data/eunis_candidate_labels.csv"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("pinned EUNIS candidate table is missing")


def _row(index: int) -> dict[str, Any]:
    sentence = f"Frozen EUNIS overlap sentence {index}."
    return {
        "source_pbf": "test-latest.osm.pbf",
        "osm_type": "way",
        "osm_id": index,
        "description_identity": f"description-{index}",
        "tag_key": "description",
        "sentence_index": 0,
        "sentence": sentence,
        "text_sha256": hashlib.sha256(sentence.encode("utf-8")).hexdigest(),
        "language_code": "eng",
        "eunis_code": "T11",
        "eunis_name": "Temperate forest",
    }


def _write_frozen_run(run_dir: Path, size: int = 100) -> list[dict[str, Any]]:
    candidate_path = _candidate_file()
    with candidate_path.open(encoding="utf-8", newline="") as stream:
        candidate_rows = list(csv.DictReader(stream))
    t11_name = next(row["eunis_name"] for row in candidate_rows if row["eunis_code"] == "T11")
    source_rows = [_row(index) for index in range(1, size + 1)]
    for row in source_rows:
        row["eunis_name"] = t11_name
    selected = select_distinct_sample(source_rows, size=size, seed=42)
    shutil.copyfile(candidate_path, run_dir / "candidate_labels.csv")
    ids = [row["sample_id"] for row in selected]
    sample = {
        "format_version": 1,
        "source": {
            "dataset": OVERLAP_DATASET,
            "revision": OVERLAP_REVISION,
            "file": "overlap.parquet",
            "sha256": OVERLAP_PARQUET_SHA256,
            "license": "OpenStreetMap ODbL-1.0; European Environment Agency CC-BY-4.0",
        },
        "source_coverage": {"row_count": 224789},
        "selection": {
            "seed": 42,
            "sample_size": size,
            "sample_ids_sha256": runner._sha256_json(ids),
            "unique_sentence_hash_count": size,
            "unique_polygon_count": size,
        },
        "candidate_labels": {
            "file": "candidate_labels.csv",
            "sha256": CANDIDATE_LABELS_SHA256,
            "count": len(candidate_rows),
            "codes": [row["eunis_code"] for row in candidate_rows],
            "text_method": "Exact pinned EUNIS English name, newline, exact EEA description.",
        },
        "selected_rows": selected,
    }
    runner._write_json_exclusive(run_dir / "frozen_sample.json", sample)
    return selected


class _PromptTokenizer:
    chat_template = "template-v1"

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.template_kwargs: list[dict[str, Any]] = []

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        self.prompts.append(messages[0]["content"])
        self.template_kwargs.append(kwargs)
        return f"rendered:{messages[0]['content']}"

    def __call__(self, rendered: str, **kwargs: Any) -> dict[str, list[int]]:
        assert kwargs == {"add_special_tokens": False}
        return {"input_ids": list(range(len(rendered.split()) + 1))}


class _FakeEngine:
    version = "0.5.20"

    def __init__(self) -> None:
        self.calls: list[list[int]] = []
        self.shutdown_called = False

    async def generate(self, input_ids: list[int]) -> dict[str, Any]:
        self.calls.append(input_ids)
        index = len(self.calls)
        if index == 1:
            raw, finish = "</think>T11", "stop"
        elif index == 2:
            raw, finish = "reasoning </think>NOT_A_CODE", "stop"
        else:
            raw, finish = "</think>U62", "length"
        return {
            "text": raw,
            "meta_info": {
                "completion_tokens": 17,
                "finish_reason": {"type": finish},
                "spec_num_correct_drafts": 5,
                "spec_num_proposed_drafts": 9,
            },
        }

    def shutdown(self) -> None:
        self.shutdown_called = True


def test_dspark_protocol_pins_target_draft_runtime_and_generation() -> None:
    assert dspark.TARGET_MODEL == "LiquidAI/LFM2.5-2.6B"
    assert dspark.TARGET_REVISION == "654f9463ce32b05d0429d76fe1f580b27d4c1ac0"
    assert dspark.DRAFT_MODEL == "LiquidAI/LFM2.5-2.6B-DSpark"
    assert dspark.DRAFT_REVISION == "458cedab07d0f7b2b05700c77e1aa463d43d6f04"
    assert dspark.SGLANG_VERSION == "0.5.20"
    assert dspark.SAMPLING == {"temperature": 0.0, "max_new_tokens": 4096}
    assert dspark.ENGINE_ARGS["speculative_algorithm"] == "DSPARK"
    assert dspark.ENGINE_ARGS["speculative_draft_attention_backend"] == "flashinfer"
    assert dspark.RUNTIME_CONTEXT_TOKENS == 128_000
    assert dspark.EXPECTED_SAMPLE_IDS_SHA256 == (
        "74ab5826b51806947215b0e1635f173ce99af13577e41a431c263cd6a8e57e72"
    )


def test_prompt_includes_sentence_and_every_defined_candidate_not_gold() -> None:
    candidates = [
        {
            "eunis_code": "T11",
            "eunis_name": "Temperate forest",
            "candidate_text": "Temperate forest\nWoodland definition.",
        },
        {
            "eunis_code": "U62",
            "eunis_name": "Tall-helophyte bed",
            "candidate_text": "Tall-helophyte bed\nWetland definition.",
        },
    ]

    prompt = dspark.build_prompt("A woodland sentence.", candidates)

    assert "A woodland sentence." in prompt
    assert '"T11"' in prompt
    assert "Woodland definition." in prompt
    assert '"U62"' in prompt
    assert "Wetland definition." in prompt
    assert "gold_eunis_code" not in prompt
    assert "E5" not in prompt


def test_prompt_rejects_empty_or_duplicate_candidate_vocabularies() -> None:
    with pytest.raises(ValueError, match="sentence must be a non-empty string"):
        dspark.build_prompt("", [{"eunis_code": "T11", "candidate_text": "forest"}])
    with pytest.raises(ValueError, match="candidate codes must be non-empty and unique"):
        dspark.build_prompt("sentence", [])
    with pytest.raises(ValueError, match="candidate codes must be non-empty and unique"):
        dspark.build_prompt(
            "sentence",
            [
                {"eunis_code": "T11", "candidate_text": "a"},
                {"eunis_code": "T11", "candidate_text": "b"},
            ],
        )


def test_chat_encoder_matches_pinned_template_kwargs_and_no_auto_special_tokens() -> None:
    tokenizer = _PromptTokenizer()

    input_ids = dspark.encode_prompt(tokenizer, "Choose one.")

    assert input_ids == [0, 1, 2]
    assert tokenizer.template_kwargs == [
        {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    ]


def test_context_guard_counts_generation_tokens_against_sglang_limit() -> None:
    dspark.validate_context_length(128_000 - 4096, max_new_tokens=4096)

    with pytest.raises(ValueError, match="prompt plus generation cap exceeds SGLang context"):
        dspark.validate_context_length(128_000 - 4095, max_new_tokens=4096)
    with pytest.raises(ValueError, match="prompt and generation token counts must be positive"):
        dspark.validate_context_length(0)


@pytest.mark.parametrize(
    ("raw", "finish", "expected_code", "status", "error"),
    [
        ("reasoning mentions T11 </think>U62\n", "stop", "U62", "valid", None),
        ("reasoning </think>NOT_A_CODE", "stop", None, "invalid", "unknown_code"),
        ("</think>T11", "length", None, "truncated", "generation_length"),
        ("reasoning only", "stop", None, "invalid", "missing_think_close"),
        ("</think>choose between T11 and U62", "stop", None, "invalid", "invalid_format"),
        ("</think>`T11`<|im_end|>", "stop", "T11", "valid", None),
        ("</think><|endoftext|>", "stop", None, "invalid", "empty_answer"),
    ],
)
def test_parser_preserves_strict_allowed_label_validation(
    raw: str, finish: str, expected_code: str | None, status: str, error: str | None
) -> None:
    parsed = dspark.parse_label(raw, finish_reason=finish, candidate_codes=["T11", "U62"])

    assert parsed.code == expected_code
    assert parsed.status == status
    assert parsed.error == error


def test_runner_uses_only_pinned_frozen_rows_and_writes_sidecar_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "pilot"
    run_dir.mkdir()
    output_dir = tmp_path / "dspark"
    selected = _write_frozen_run(run_dir)
    frozen_before = (run_dir / "frozen_sample.json").read_bytes()
    (run_dir / "predictions.parquet").write_bytes(b"existing E5 predictions")
    tokenizer = _PromptTokenizer()
    engine = _FakeEngine()
    monkeypatch.setattr(dspark, "load_tokenizer", lambda: tokenizer)
    monkeypatch.setattr(dspark, "SGLangEngine", lambda *_args: engine)
    monkeypatch.setattr(dspark_runner, "_require_supported_gpu", lambda: {"name": "mock GPU"})
    monkeypatch.setattr(dspark_runner, "_prepare_model_cache", lambda _path: tmp_path / "cache")
    monkeypatch.setattr(dspark, "RUNTIME_CONTEXT_TOKENS", 128_000)
    monkeypatch.setattr(
        dspark,
        "EXPECTED_SAMPLE_IDS_SHA256",
        runner._sha256_json([row["sample_id"] for row in selected]),
    )
    monkeypatch.setattr(dspark_runner, "_runtime_metadata", lambda: {"python": "3.12.0"})
    monkeypatch.setattr(dspark_runner, "_clock", iter(range(10_000)).__next__)

    result = dspark_runner.run_dspark_pilot(
        run_dir,
        computation_commit="a" * 40,
        validation_commit="b" * 40,
        output_dir=output_dir,
    )

    predictions = pq.read_table(output_dir / "dspark_predictions.parquet").to_pylist()
    metrics = json.loads((output_dir / "dspark_metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "dspark_manifest.json").read_text(encoding="utf-8"))
    assert result["sample_count"] == 100
    assert result["run_dir"] == str(output_dir)
    assert len(predictions) == 100
    assert len(engine.calls) == 100
    assert engine.shutdown_called
    assert len(tokenizer.prompts) == 100
    assert predictions[0]["raw_output"] == "</think>T11"
    assert predictions[0]["parsed_eunis_code"] == "T11"
    assert predictions[0]["parse_status"] == "valid"
    assert predictions[1]["raw_output"] == "reasoning </think>NOT_A_CODE"
    assert predictions[1]["parsed_eunis_code"] is None
    assert predictions[1]["parse_error"] == "unknown_code"
    assert predictions[2]["parse_status"] == "truncated"
    assert predictions[2]["parsed_eunis_code"] is None
    assert metrics["overall"]["sample_count"] == 100
    assert metrics["overall"]["prediction_count"] == 1
    assert metrics["overall"]["invalid_output_count"] == 99
    assert manifest["model"]["revision"] == dspark.TARGET_REVISION
    assert manifest["draft"]["revision"] == dspark.DRAFT_REVISION
    assert manifest["sample"]["sample_ids_sha256"] == runner._sha256_json(
        [row["sample_id"] for row in selected]
    )
    assert manifest["outputs_sha256"]["dspark_predictions.parquet"]
    assert (run_dir / "frozen_sample.json").read_bytes() == frozen_before
    assert (run_dir / "predictions.parquet").read_bytes() == b"existing E5 predictions"


def test_runner_rejects_a_different_sample_before_loading_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "pilot"
    run_dir.mkdir()
    _write_frozen_run(run_dir)
    monkeypatch.setattr(dspark, "EXPECTED_SAMPLE_IDS_SHA256", "0" * 64)
    monkeypatch.setattr(dspark, "load_tokenizer", lambda: pytest.fail("must reject before load"))

    with pytest.raises(ValueError, match="frozen sample IDs do not match the published pilot"):
        dspark_runner.run_dspark_pilot(
            run_dir,
            computation_commit="a" * 40,
            validation_commit="b" * 40,
        )


@pytest.mark.parametrize(
    ("selection_key", "selection_value", "candidate_hash", "message"),
    [
        ("sample_size", 99, CANDIDATE_LABELS_SHA256, "requires the published 100-row pilot"),
        ("seed", 41, CANDIDATE_LABELS_SHA256, "sample seed does not match"),
        ("sample_size", 100, "0" * 64, "candidate CSV hash does not match"),
    ],
)
def test_runner_rejects_sample_protocol_drift(
    selection_key: str,
    selection_value: int,
    candidate_hash: str,
    message: str,
) -> None:
    sample = {
        "selection": {
            "sample_size": 100,
            "seed": 42,
            "sample_ids_sha256": dspark.EXPECTED_SAMPLE_IDS_SHA256,
        },
        "candidate_labels": {"sha256": candidate_hash},
    }
    sample["selection"][selection_key] = selection_value

    with pytest.raises(ValueError, match=message):
        dspark_runner._validate_published_sample(sample, [{}] * 100)


def test_runner_rejects_context_overflow_before_constructing_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "pilot"
    run_dir.mkdir()
    selected = _write_frozen_run(run_dir)
    monkeypatch.setattr(
        dspark,
        "EXPECTED_SAMPLE_IDS_SHA256",
        runner._sha256_json([row["sample_id"] for row in selected]),
    )
    monkeypatch.setattr(dspark, "load_tokenizer", lambda: _PromptTokenizer())
    monkeypatch.setattr(dspark_runner, "_require_supported_gpu", lambda: {"name": "mock GPU"})
    monkeypatch.setattr(dspark_runner, "_prepare_model_cache", lambda _path: tmp_path / "cache")
    monkeypatch.setattr(
        dspark,
        "encode_prompt",
        lambda *_args: [0] * (dspark.RUNTIME_CONTEXT_TOKENS - dspark.MAX_NEW_TOKENS + 1),
    )
    monkeypatch.setattr(
        dspark,
        "SGLangEngine",
        lambda *_args: pytest.fail("context must be checked before engine load"),
    )

    with pytest.raises(ValueError, match="prompt plus generation cap exceeds SGLang context"):
        dspark_runner.run_dspark_pilot(
            run_dir,
            computation_commit="a" * 40,
            validation_commit="b" * 40,
        )


def test_runner_refuses_to_overwrite_an_existing_dspark_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "pilot"
    run_dir.mkdir()
    output_dir = tmp_path / "dspark"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="DSpark output directory already exists"):
        dspark_runner.run_dspark_pilot(
            run_dir,
            computation_commit="a" * 40,
            validation_commit="b" * 40,
            output_dir=output_dir,
        )


def test_shared_single_label_metrics_count_invalid_outputs_as_incorrect() -> None:
    from georeset_text_label_benchmark.pilot.metrics import single_label_report

    result = single_label_report(["A", "B", "C"], ["A", None, "B"], ["A", "B", "C"])

    assert result["sample_count"] == 3
    assert result["prediction_count"] == 2
    assert result["invalid_output_count"] == 1
    assert result["coverage"] == pytest.approx(2 / 3)
    assert result["top1_accuracy"] == pytest.approx(1 / 3)
    assert result["macro_f1_all_candidates"] == pytest.approx(1 / 3)


def test_sglang_engine_adapter_forwards_exact_generation_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}

    class _StubSGLang:
        __version__ = "0.5.20"

        class Engine:
            def __init__(self, **kwargs: Any) -> None:
                calls["engine_kwargs"] = kwargs

            async def async_generate(self, **kwargs: Any) -> dict[str, Any]:
                calls["generation_kwargs"] = kwargs
                return {"text": "</think>T11"}

            def shutdown(self) -> None:
                calls["shutdown"] = True

    monkeypatch.setitem(__import__("sys").modules, "sglang", _StubSGLang)
    engine = dspark.SGLangEngine(dspark.engine_kwargs(), dspark.SAMPLING)
    result = asyncio.run(engine.generate([1, 2, 3]))
    engine.shutdown()

    assert result == {"text": "</think>T11"}
    assert calls["engine_kwargs"]["model_path"] == dspark.TARGET_MODEL
    assert calls["engine_kwargs"]["revision"] == dspark.TARGET_REVISION
    assert calls["engine_kwargs"]["speculative_draft_model_path"] == dspark.DRAFT_MODEL
    assert calls["generation_kwargs"] == {
        "input_ids": [1, 2, 3],
        "sampling_params": dspark.SAMPLING,
    }
    assert calls["shutdown"] is True


def test_prompt_and_chat_template_hashes_are_pinned_for_manifest() -> None:
    tokenizer = _PromptTokenizer()

    assert dspark.template_sha256(tokenizer) == hashlib.sha256(b"template-v1").hexdigest()
    expected = hashlib.sha256(dspark.PROMPT_INSTRUCTIONS.encode()).hexdigest()
    assert expected == dspark.PROMPT_SHA256


def test_tokenizer_loader_uses_the_pinned_model_and_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> str:
            calls["args"] = args
            calls["kwargs"] = kwargs
            return "mock tokenizer"

    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=_AutoTokenizer),
    )

    assert dspark.load_tokenizer() == "mock tokenizer"
    assert calls == {
        "args": (dspark.TARGET_MODEL,),
        "kwargs": {"revision": dspark.TARGET_REVISION, "trust_remote_code": False},
    }


def test_generation_result_metadata_accepts_string_and_missing_finish_reasons() -> None:
    assert dspark.output_finish_reason({"meta_info": {"finish_reason": "length"}}) == "length"
    assert dspark.output_finish_reason({}) == "unknown"
    assert dspark.output_metadata({"meta_info": {"completion_tokens": 3}}) == {
        "completion_tokens": 3
    }
    assert dspark.output_metadata({"meta_info": None}) == {}
    assert dspark_runner._optional_int(None) is None
    assert dspark_runner._optional_int("5") == 5


def test_gpu_adapter_checks_visibility_before_querying_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dspark_runner.shutil, "which", lambda _name: None)

    with pytest.raises(RuntimeError, match="visible NVIDIA CUDA GPU and nvidia-smi"):
        dspark_runner._require_supported_gpu()


def test_gpu_adapter_records_a_supported_visible_device(monkeypatch: pytest.MonkeyPatch) -> None:
    completed = SimpleNamespace(stdout="H100, 81920, 9.0\n")
    monkeypatch.setattr(dspark_runner.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(dspark_runner.subprocess, "run", lambda *_args, **_kwargs: completed)

    assert dspark_runner._require_supported_gpu() == {
        "name": "H100",
        "memory_mib": 81920,
        "compute_capability": "9.0",
    }


def test_model_cache_defaults_to_selected_path_and_requires_eight_gib_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    cache = tmp_path / "models"
    monkeypatch.setattr(
        dspark_runner.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=dspark_runner.MIN_CACHE_FREE_BYTES),
    )

    assert dspark_runner._prepare_model_cache(cache) == (cache / "hub").resolve()
    assert Path(__import__("os").environ["HF_HOME"]) == cache.resolve()


def test_model_cache_respects_hub_override_and_fails_when_low_on_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hf_home = tmp_path / "existing-home"
    hub_cache = tmp_path / "hub-cache"
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("HF_HUB_CACHE", str(hub_cache))
    monkeypatch.setattr(
        dspark_runner.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=dspark_runner.MIN_CACHE_FREE_BYTES - 1),
    )

    with pytest.raises(OSError, match="model cache needs 8 GiB free"):
        dspark_runner._prepare_model_cache(tmp_path / "ignored")

    assert Path(__import__("os").environ["HF_HOME"]) == hf_home
    assert hub_cache.is_dir()


def test_single_label_metrics_validate_lengths_unknown_codes_and_group_nulls() -> None:
    from georeset_text_label_benchmark.pilot.metrics import (
        single_label_group_breakdown,
        single_label_report,
    )

    with pytest.raises(ValueError, match="gold and single-label prediction lengths"):
        single_label_report(["A"], [], ["A"])
    with pytest.raises(ValueError, match="unknown gold code: X"):
        single_label_report(["X"], ["A"], ["A"])
    with pytest.raises(ValueError, match="unknown prediction code: X"):
        single_label_report(["A"], ["X"], ["A"])
    with pytest.raises(ValueError, match="candidate codes must be non-empty and unique"):
        single_label_report(["A"], ["A"], ["A", "A"])

    groups = single_label_group_breakdown([None, "eng"], ["A", "B"], [None, "B"])
    assert groups == [
        {
            "group": "eng",
            "sample_count": 1,
            "prediction_count": 1,
            "coverage": 1.0,
            "top1_accuracy": 1.0,
        },
        {
            "group": "missing",
            "sample_count": 1,
            "prediction_count": 0,
            "coverage": 0.0,
            "top1_accuracy": 0.0,
        },
    ]


def test_dspark_cli_uses_frozen_run_directory_and_cache_settings(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def run(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append((args, kwargs))
        return {"sample_count": 100}

    monkeypatch.setattr(cli, "run_dspark_pilot", run)

    assert (
        cli.main(
            [
                "run-dspark",
                "--run-dir",
                "frozen-inputs",
                "--model-cache",
                "hf-cache",
                "--output-dir",
                "dspark-output",
                "--computation-commit",
                "a" * 40,
                "--validation-commit",
                "b" * 40,
            ]
        )
        == 0
    )

    assert capsys.readouterr().out == '{\n  "sample_count": 100\n}\n'
    assert calls == [
        (
            (Path("frozen-inputs"),),
            {
                "model_cache_dir": Path("hf-cache"),
                "output_dir": Path("dspark-output"),
                "computation_commit": "a" * 40,
                "validation_commit": "b" * 40,
            },
        )
    ]


def test_gpu_gate_rejects_insufficient_memory_or_compute_capability() -> None:
    with pytest.raises(RuntimeError, match="DSpark GPU gate requires"):
        dspark_runner._validate_gpu_record("old gpu, 8192, 7.5")

    assert dspark_runner._validate_gpu_record("H100, 81920, 9.0") == {
        "name": "H100",
        "memory_mib": 81920,
        "compute_capability": "9.0",
    }
