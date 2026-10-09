"""Fail the CI job when mutation results contain anything but killed mutants."""

from __future__ import annotations

import hashlib
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass

RESULT_PATTERN = re.compile(r"\s*(\S+): (killed|survived|no tests|timeout)")
EQUIVALENT_RATIONALES = {
    "georeset_text_label_benchmark.join.x__validate_text_hash__mutmut_18": (
        "In _validate_text_hash, the mutant changes False to None in an exception branch; "
        "both values are false under the only subsequent `if not valid_digest`, so every "
        "input and emitted exception are identical. Invalid hexadecimal digests are tested."
    ),
    "georeset_text_label_benchmark.join.x__verify_sentence_hash__mutmut_6": (
        "In _verify_sentence_hash, Python's codec registry resolves utf-8 and UTF-8 to the "
        "same codec and identical bytes; exact sentence SHA-256 validation is tested."
    ),
    "georeset_text_label_benchmark.pilot.cli.x_main__mutmut_50": (
        "json.dumps treats ensure_ascii=None as false, so CLI results have identical "
        "Unicode serialization to ensure_ascii=False."
    ),
    "georeset_text_label_benchmark.pilot.cli.x_main__mutmut_17": (
        "json.dumps treats ensure_ascii=None as false, so CLI results have identical "
        "Unicode serialization to ensure_ascii=False."
    ),
    "georeset_text_label_benchmark.pilot.dspark.x_build_prompt__mutmut_22": (
        "build_prompt constructs codes by iterating the same candidate sequence immediately "
        "before this zip, so both iterables always have equal lengths; strict=None cannot "
        "alter the rendered candidate list."
    ),
    "georeset_text_label_benchmark.pilot.dspark.x_build_prompt__mutmut_25": (
        "build_prompt constructs codes by iterating the same candidate sequence immediately "
        "before this zip, so both iterables always have equal lengths; omitting strict "
        "cannot alter the rendered candidate list."
    ),
    "georeset_text_label_benchmark.pilot.dspark.x_build_prompt__mutmut_26": (
        "build_prompt constructs codes by iterating the same candidate sequence immediately "
        "before this zip, so both iterables always have equal lengths; strict=False cannot "
        "alter the rendered candidate list."
    ),
    "georeset_text_label_benchmark.pilot.dspark.x_build_prompt__mutmut_29": (
        "json.dumps treats ensure_ascii=None as false, so Unicode candidate text is "
        "serialized identically to ensure_ascii=False; the prompt contract includes "
        "non-ASCII text."
    ),
    "georeset_text_label_benchmark.pilot.dspark.x_encode_prompt__mutmut_8": (
        "CHAT_TEMPLATE_KWARGS is the pinned empty mapping, so omitting its keyword "
        "expansion passes the same arguments to apply_chat_template. The template test "
        "asserts those arguments and the rendered prompt."
    ),
    "georeset_text_label_benchmark.pilot.dspark.x_template_sha256__mutmut_5": (
        "Python resolves utf-8 and UTF-8 to the same codec, producing the same pinned chat "
        "template SHA-256."
    ),
    "georeset_text_label_benchmark.pilot.dspark_runner.x__encode_frozen_prompts__mutmut_21": (
        "Python resolves utf-8 and UTF-8 to the same codec, so per-prompt SHA-256 values "
        "are unchanged."
    ),
    "georeset_text_label_benchmark.pilot.dspark_runner.x__prediction_row__mutmut_41": (
        "parse_label returns either None or a non-empty allowed code, and candidate_names "
        "has string keys only. candidate_names.get(None) is None, matching the conditional "
        "branch."
    ),
    "georeset_text_label_benchmark.pilot.dspark_runner.x__read_frozen_e5_manifest__mutmut_10": (
        "Python resolves utf-8 and UTF-8 to the same codec when reading the manifest JSON."
    ),
    "georeset_text_label_benchmark.pilot.dspark_runner.x__new_seed_content_hasher__mutmut_13": (
        "Python's codec registry resolves ascii and ASCII identically, so the Git blob header "
        "bytes and resulting SHA-1 are unchanged."
    ),
    "georeset_text_label_benchmark.pilot.dspark_runner.x__seed_content_digest__mutmut_18": (
        "dict.get without a default returns None; for an unknown digest algorithm, that "
        "preserves the same invalid-digest rejection as the explicit zero default."
    ),
    "georeset_text_label_benchmark.pilot.dspark_runner.x__seed_content_digest__mutmut_20": (
        "Omitting dict.get's default returns None; for an unknown digest algorithm, that "
        "preserves the same invalid-digest rejection as the explicit zero default."
    ),
    "georeset_text_label_benchmark.pilot.dspark_runner.x__seed_content_digest__mutmut_30": (
        "The supported digest lengths are only 40 and 64, while an unknown algorithm maps to "
        "zero; replacing the zero comparison with one preserves all validation outcomes."
    ),
    "georeset_text_label_benchmark.pilot.dspark_runner.x__seed_snapshot_entry__mutmut_63": (
        "_check_seed_blob_collision rejects conflicting source paths before "
        "_copy_seed_blob writes the already-checked mapping."
    ),
    "georeset_text_label_benchmark.pilot.dspark_runner.x__seed_snapshot_entry__mutmut_64": (
        "_check_seed_blob_collision rejects conflicting source paths before "
        "_copy_seed_blob writes the already-checked mapping."
    ),
    "georeset_text_label_benchmark.pilot.dspark_runner.x_run_dspark_pilot__mutmut_117": (
        "PyArrow treats zstd and ZSTD as aliases for the same Parquet codec; the adapter "
        "test checks that the emitted metadata reports ZSTD."
    ),
    "georeset_text_label_benchmark.pilot.dspark_runner.x_run_dspark_pilot__mutmut_140": (
        "PyArrow treats zstd and ZSTD as aliases for the same Parquet codec; the adapter "
        "test checks that the emitted metadata reports ZSTD."
    ),
    "georeset_text_label_benchmark.pilot.dspark_runner.x_run_dspark_pilot__mutmut_145": (
        "PyArrow treats zstd and ZSTD as aliases for the same Parquet codec; the adapter "
        "test checks that the emitted metadata reports ZSTD."
    ),
    "georeset_text_label_benchmark.pilot.embeddings.x_average_pool__mutmut_28": (
        "Tensor.unsqueeze accepts +1 and 1 as the same dimension index."
    ),
    "georeset_text_label_benchmark.pilot.embeddings.x_encode_texts__mutmut_46": (
        "For torch.nn.functional.normalize with dim=1, p=None and p=2 select the same "
        "vector norm; a non-unit [3, 4] vector is checked against [0.6, 0.8]."
    ),
    "georeset_text_label_benchmark.pilot.embeddings.x_encode_texts__mutmut_49": (
        "torch.nn.functional.normalize defaults p to 2, so omitting p preserves the "
        "explicit p=2 result."
    ),
    "georeset_text_label_benchmark.pilot.embeddings.x_encode_texts__mutmut_50": (
        "torch.nn.functional.normalize defaults dim to 1, so omitting dim preserves the "
        "explicit dim=1 result."
    ),
    "georeset_text_label_benchmark.pilot.embeddings.x_encode_texts__mutmut_56": (
        "torch.cat defaults dim to 0, so omitting dim preserves concatenation along the batch axis."
    ),
    "georeset_text_label_benchmark.pilot.embeddings.x_rank_candidates__mutmut_21": (
        "_validate_embedding_shapes requires the candidate row count to equal the code "
        "count, so score rows and codes always have equal lengths before this zip."
    ),
    "georeset_text_label_benchmark.pilot.embeddings.x_rank_candidates__mutmut_24": (
        "_validate_embedding_shapes requires the candidate row count to equal the code "
        "count, so omitting strict cannot change this zip's aligned iteration."
    ),
    "georeset_text_label_benchmark.pilot.embeddings.x_rank_candidates__mutmut_25": (
        "_validate_embedding_shapes requires the candidate row count to equal the code "
        "count, so strict=False cannot change this zip's aligned iteration."
    ),
    "georeset_text_label_benchmark.pilot.metrics.x__validate_predictions__mutmut_27": (
        "_validate_predictions checks all three sequence lengths before this zip; replacing "
        "strict=True with strict=None cannot change iteration or validation."
    ),
    "georeset_text_label_benchmark.pilot.metrics.x__validate_predictions__mutmut_30": (
        "_validate_predictions checks all three sequence lengths before this zip; omitting "
        "strict therefore iterates the same aligned values."
    ),
    "georeset_text_label_benchmark.pilot.metrics.x__validate_predictions__mutmut_31": (
        "_validate_predictions checks all three sequence lengths before this zip; "
        "strict=False therefore iterates the same aligned values."
    ),
    "georeset_text_label_benchmark.pilot.metrics.x_class_breakdown__mutmut_20": (
        "class_breakdown calls _validate_predictions, which enforces equal lengths before "
        "this zip; strict=None has identical results."
    ),
    "georeset_text_label_benchmark.pilot.metrics.x_class_breakdown__mutmut_23": (
        "class_breakdown calls _validate_predictions, which enforces equal lengths before "
        "this zip; omitting strict has identical results."
    ),
    "georeset_text_label_benchmark.pilot.metrics.x_class_breakdown__mutmut_24": (
        "class_breakdown calls _validate_predictions, which enforces equal lengths before "
        "this zip; strict=False has identical results."
    ),
    "georeset_text_label_benchmark.pilot.metrics.x_single_label_report__mutmut_15": (
        "single_label_report validates equal non-empty gold and prediction lengths before "
        "this zip, so strict=None has the same aligned iteration."
    ),
    "georeset_text_label_benchmark.pilot.metrics.x_single_label_report__mutmut_18": (
        "single_label_report validates equal non-empty gold and prediction lengths before "
        "this zip, so omitting strict has the same aligned iteration."
    ),
    "georeset_text_label_benchmark.pilot.metrics.x_single_label_report__mutmut_19": (
        "single_label_report validates equal non-empty gold and prediction lengths before "
        "this zip, so strict=False has the same aligned iteration."
    ),
    "georeset_text_label_benchmark.pilot.publication.x__probe_exclusive_link__mutmut_15": (
        "The probe writes a regular source file and creates a regular hard link immediately "
        "before these stat calls, so the follow flag cannot change their inode/device "
        "identity. Separate publication tests cover symlinks and hard links to symlinks."
    ),
    "georeset_text_label_benchmark.pilot.publication.x__probe_exclusive_link__mutmut_16": (
        "The probe writes a regular source file and creates a regular hard link immediately "
        "before these stat calls, so the follow flag cannot change their inode/device "
        "identity. Separate publication tests cover symlinks and hard links to symlinks."
    ),
    "georeset_text_label_benchmark.pilot.publication.x__probe_exclusive_link__mutmut_18": (
        "The probe writes a regular source file and creates a regular hard link immediately "
        "before these stat calls, so the follow flag cannot change their inode/device "
        "identity. Separate publication tests cover symlinks and hard links to symlinks."
    ),
    "georeset_text_label_benchmark.pilot.publication.x__probe_exclusive_link__mutmut_19": (
        "The probe writes a regular source file and creates a regular hard link immediately "
        "before these stat calls, so the follow flag cannot change their inode/device "
        "identity. Separate publication tests cover symlinks and hard links to symlinks."
    ),
    "georeset_text_label_benchmark.pilot.publication.x_ensure_publication_supported__mutmut_14": (
        "This mutation changes only the private capability probe name or payload. The probe "
        "still runs under the destination parent, verifies an exclusive hard link and inode "
        "identity, and removes its temporary files."
    ),
    "georeset_text_label_benchmark.pilot.publication.x_ensure_publication_supported__mutmut_15": (
        "This mutation changes only the private capability probe name or payload. The probe "
        "still runs under the destination parent, verifies an exclusive hard link and inode "
        "identity, and removes its temporary files."
    ),
    "georeset_text_label_benchmark.pilot.publication.x_ensure_publication_supported__mutmut_18": (
        "This mutation changes only the private capability probe name or payload. The probe "
        "still runs under the destination parent, verifies an exclusive hard link and inode "
        "identity, and removes its temporary files."
    ),
    "georeset_text_label_benchmark.pilot.publication.x_ensure_publication_supported__mutmut_19": (
        "This mutation changes only the private capability probe name or payload. The probe "
        "still runs under the destination parent, verifies an exclusive hard link and inode "
        "identity, and removes its temporary files."
    ),
    "georeset_text_label_benchmark.pilot.publication.x_ensure_publication_supported__mutmut_21": (
        "This mutation changes only the private capability probe name or payload. The probe "
        "still runs under the destination parent, verifies an exclusive hard link and inode "
        "identity, and removes its temporary files."
    ),
    "georeset_text_label_benchmark.pilot.publication.x_ensure_publication_supported__mutmut_22": (
        "This mutation changes only the private capability probe name or payload. The probe "
        "still runs under the destination parent, verifies an exclusive hard link and inode "
        "identity, and removes its temporary files."
    ),
    "georeset_text_label_benchmark.pilot.publication.x_ensure_publication_supported__mutmut_4": (
        "This mutation changes only the private capability probe name or payload. The probe "
        "still runs under the destination parent, verifies an exclusive hard link and inode "
        "identity, and removes its temporary files."
    ),
    "georeset_text_label_benchmark.pilot.publication.x_ensure_publication_supported__mutmut_6": (
        "This mutation changes only the private capability probe name or payload. The probe "
        "still runs under the destination parent, verifies an exclusive hard link and inode "
        "identity, and removes its temporary files."
    ),
    "georeset_text_label_benchmark.pilot.publication.x_ensure_publication_supported__mutmut_8": (
        "This mutation changes only the private capability probe name or payload. The probe "
        "still runs under the destination parent, verifies an exclusive hard link and inode "
        "identity, and removes its temporary files."
    ),
    "georeset_text_label_benchmark.pilot.publication.x_ensure_publication_supported__mutmut_9": (
        "This mutation changes only the private capability probe name or payload. The probe "
        "still runs under the destination parent, verifies an exclusive hard link and inode "
        "identity, and removes its temporary files."
    ),
    "georeset_text_label_benchmark.pilot.runner.x__read_frozen_sample__mutmut_7": (
        "Python resolves UTF-8 and utf-8 to the same codec; the reader still decodes the "
        "same bytes. Tests require an explicit UTF-8 codec and exact filename."
    ),
    "georeset_text_label_benchmark.pilot.runner.x__runtime_metadata__mutmut_10": (
        "importlib.metadata normalizes distribution names case-insensitively; "
        "version('TORCH') returns the same installed version as version('torch')."
    ),
    "georeset_text_label_benchmark.pilot.runner.x__runtime_metadata__mutmut_15": (
        "importlib.metadata normalizes distribution names case-insensitively; "
        "version('TRANSFORMERS') returns the same installed version as "
        "version('transformers')."
    ),
    "georeset_text_label_benchmark.pilot.runner.x__runtime_metadata__mutmut_20": (
        "importlib.metadata normalizes distribution names case-insensitively; "
        "version('PYARROW') returns the same installed version as version('pyarrow')."
    ),
    "georeset_text_label_benchmark.pilot.runner.x__sha256_json__mutmut_17": (
        "Python resolves utf-8 and UTF-8 to the same codec, so encoding the canonical JSON "
        "payload produces identical bytes and hashes."
    ),
    "georeset_text_label_benchmark.pilot.runner.x__sha256_json__mutmut_3": (
        "json.dumps treats ensure_ascii=None as false, so it emits the same text as "
        "ensure_ascii=False for every JSON value."
    ),
    "georeset_text_label_benchmark.pilot.runner.x__write_json_exclusive__mutmut_14": (
        "json.dump treats ensure_ascii=None as false, so its UTF-8 JSON bytes match the "
        "ensure_ascii=False output."
    ),
    "georeset_text_label_benchmark.pilot.runner.x__write_outputs__mutmut_14": (
        "PyArrow treats zstd and ZSTD as aliases for the same Parquet codec; a metadata "
        "assertion checks the resulting compression."
    ),
    "georeset_text_label_benchmark.pilot.runner.x_freeze_sample__mutmut_32": (
        "Path.mkdir tests exist_ok by truth value; None and False both reject an existing "
        "directory. A race regression verifies that True is rejected."
    ),
    "georeset_text_label_benchmark.pilot.runner.x_freeze_sample__mutmut_34": (
        "Path.mkdir defaults exist_ok to False, so omitting the explicit False preserves "
        "the same exclusive directory creation behavior."
    ),
    "georeset_text_label_benchmark.pilot.runner.x_run_pilot__mutmut_98": (
        "rank_candidates defaults top_k to 5; omitting the explicit top_k=5 argument "
        "preserves the same ranking size, and output rows are verified."
    ),
    "georeset_text_label_benchmark.pilot.runner.x_sha256_file__mutmut_9": (
        "hashlib accepts sha256 and SHA256 as case-insensitive algorithm names; the digest "
        "is identical for the same file bytes."
    ),
    "georeset_text_label_benchmark.pilot.sampling.x__rank_rows__mutmut_5": (
        "Python resolves ascii and ASCII to the same codec; the deterministic ranking "
        "digest is unchanged."
    ),
    "georeset_text_label_benchmark.pilot.sampling.x__sample_id__mutmut_11": (
        "The sample identity payload is always a JSON list, which has no key/value "
        "separator; changing the unused colon separator cannot change its serialized bytes."
    ),
    "georeset_text_label_benchmark.pilot.sampling.x__sample_id__mutmut_15": (
        "Python resolves utf-8 and UTF-8 to the same codec, so stable sample identity "
        "hashes are unchanged."
    ),
    "georeset_text_label_benchmark.pilot.sampling.x__sample_id__mutmut_4": (
        "json.dumps treats ensure_ascii=None as false, so the stable identity JSON bytes "
        "match ensure_ascii=False."
    ),
    "georeset_text_label_benchmark.pilot.sampling.x__validate_sentence_hash__mutmut_5": (
        "Python resolves utf-8 and UTF-8 to the same codec, so the sentence bytes and "
        "SHA-256 are unchanged."
    ),
    "georeset_text_label_benchmark.pipeline.x__build_run__mutmut_21": (
        "In _build_run, PyArrow accepts zstd and ZSTD as the same compression codec; the "
        "produced Parquet metadata is verified to report ZSTD."
    ),
    "georeset_text_label_benchmark.quality.mutation.x__mutation_fingerprint__mutmut_20": (
        "Python resolves UTF-8 and utf-8 to the same codec, so encoding the canonical patch "
        "text produces the same fingerprint bytes."
    ),
    "georeset_text_label_benchmark.pilot.cli.x_main__mutmut_13": (
        "json.dumps treats ensure_ascii=None as false, so CLI results have identical Unicode serialization to ensure_ascii=False."
    ),
    "georeset_text_label_benchmark.pilot.metrics.x__true_positives__mutmut_4": (
        "_true_positives runs only after single-label validation checks equal gold and prediction lengths, so strict=None has the same aligned iteration."
    ),
    "georeset_text_label_benchmark.pilot.metrics.x__true_positives__mutmut_7": (
        "_true_positives runs only after single-label validation checks equal gold and prediction lengths, so omitting strict has the same aligned iteration."
    ),
    "georeset_text_label_benchmark.pilot.metrics.x__true_positives__mutmut_8": (
        "_true_positives runs only after single-label validation checks equal gold and prediction lengths, so strict=False has the same aligned iteration."
    ),
}


@dataclass(frozen=True)
class MutationExemption:
    """Reviewed evidence for one exact surviving mutant patch."""

    fingerprint: str
    rationale: str


EQUIVALENT_FINGERPRINTS: dict[str, str] = {
    "georeset_text_label_benchmark.join.x__validate_text_hash__mutmut_18": "c3b9b40fd4e77b6cd037e55b3e755087df8532ce6d1909c311c01dba11913cf5",
    "georeset_text_label_benchmark.join.x__verify_sentence_hash__mutmut_6": "897e9d2066e18847fa513aba786d30d3400d297aa6e0bfd89ae586daeff87be4",
    "georeset_text_label_benchmark.pilot.cli.x_main__mutmut_50": "e317e803cb4a0b9adac8aa69a4db7090b923ebd5e75956ec4f7bf73e013419f4",
    "georeset_text_label_benchmark.pilot.cli.x_main__mutmut_17": "fbd9a3a182d4af3531d16921fd819b6a0f11ed3f0e18a0b01cb7d86dcffb9f14",
    "georeset_text_label_benchmark.pilot.dspark.x_build_prompt__mutmut_22": "66e4967360377605713309559bad5ba7744f86f6543608975a63392e55aabee5",
    "georeset_text_label_benchmark.pilot.dspark.x_build_prompt__mutmut_25": "63f0450d9da1af7695b5194c67202188a4a6056fdeba50bbf62728ff0e4f2ec9",
    "georeset_text_label_benchmark.pilot.dspark.x_build_prompt__mutmut_26": "70760f50df9058bf4d75191b5da89c349db7d913da18cb8f90105be5370ec6ed",
    "georeset_text_label_benchmark.pilot.dspark.x_build_prompt__mutmut_29": "1e7e3b34a51a53dd00e355287bf8e6da7842a8aac46e2361f43d69d3a3b27cdd",
    "georeset_text_label_benchmark.pilot.dspark.x_encode_prompt__mutmut_8": "e20685cb5e6d61ee03d6af90bb9deddb91012c9823f7d042c39ff8043d18bc50",
    "georeset_text_label_benchmark.pilot.dspark.x_template_sha256__mutmut_5": "ca029cc80aaa0f720c127835532cf1666b1ea025cc3882ab3a8b1dfc057542d2",
    "georeset_text_label_benchmark.pilot.dspark_runner.x__encode_frozen_prompts__mutmut_21": "ac9d8036932a65cafca518c618417eda57e1a87b9e26c6ba91c35698124848cc",
    "georeset_text_label_benchmark.pilot.dspark_runner.x__prediction_row__mutmut_41": "f567c307849498199c56c0b5b5fd47aa914f613e516c8618e0451837e9194c56",
    "georeset_text_label_benchmark.pilot.dspark_runner.x__read_frozen_e5_manifest__mutmut_10": "5ad66d9af22bf0a6d8490a046db51195f17c4466a78e7b5968f6818cc2db273f",
    "georeset_text_label_benchmark.pilot.dspark_runner.x__new_seed_content_hasher__mutmut_13": "a12617dcc07e8cf3a7d11719897753ba26a9e35fcdca2ed13f796345253268ee",
    "georeset_text_label_benchmark.pilot.dspark_runner.x__seed_content_digest__mutmut_18": "218b57232dd04977f43fbaf2ee9e418982aa8fb48f52773337c675b2ab29d79e",
    "georeset_text_label_benchmark.pilot.dspark_runner.x__seed_content_digest__mutmut_20": "8857c05ed7d312df9c4c89a68d223ceff53e78ea23890ad4c748b78df663ade9",
    "georeset_text_label_benchmark.pilot.dspark_runner.x__seed_content_digest__mutmut_30": "1f560e886c54fbebc1da628bc728a6ca876e8cc9b711797646b8ffe90e43db17",
    "georeset_text_label_benchmark.pilot.dspark_runner.x__seed_snapshot_entry__mutmut_63": "0bfd009d988450f6deb0484b26f11e169cc4858aaf9486b8e664cb9bc9bd644d",
    "georeset_text_label_benchmark.pilot.dspark_runner.x__seed_snapshot_entry__mutmut_64": "0a9fb48c9da4ce556c52551b3c944a3a381dbe05f8dbbc5e23928b83c49d1fbb",
    "georeset_text_label_benchmark.pilot.dspark_runner.x_run_dspark_pilot__mutmut_117": "e045243688cc4d3e63a821de04a801070bb4f6649ff06a58de31dbcd927cc354",
    "georeset_text_label_benchmark.pilot.dspark_runner.x_run_dspark_pilot__mutmut_140": "ade27ff44bf497cf2c7beac3cb6b21c902f7e33da3797d4a266b22487367a016",
    "georeset_text_label_benchmark.pilot.dspark_runner.x_run_dspark_pilot__mutmut_145": "719c04a01f20727321e10b97d631e4ba11731a59476562e980df6e7ec926efcd",
    "georeset_text_label_benchmark.pilot.embeddings.x_average_pool__mutmut_28": "6df3a896f6cea2f72dd0eab8273ac2447f1f8ad7219bc656d651e728ec548f6a",
    "georeset_text_label_benchmark.pilot.embeddings.x_encode_texts__mutmut_46": "762fcd4391bf4f011b03d439abcd817c23b83fdbc386a50c5b090ce5363bf529",
    "georeset_text_label_benchmark.pilot.embeddings.x_encode_texts__mutmut_49": "52e595bb1289043f8ddb669f2998e8ad247382744d3a255ce445a728f3a32d84",
    "georeset_text_label_benchmark.pilot.embeddings.x_encode_texts__mutmut_50": "c0836a195870fef56559604dd4ddaacc11938b89b139b12e005cedcae9129c77",
    "georeset_text_label_benchmark.pilot.embeddings.x_encode_texts__mutmut_56": "7e859a3e3a41c180568c5e1e68ba59aaea7fea947686331e64a5ea0df64f9615",
    "georeset_text_label_benchmark.pilot.embeddings.x_rank_candidates__mutmut_21": "8202d36d7497fba64829bd04edff5ce43afe22eb49dd132d2f69d63acc6c2a7a",
    "georeset_text_label_benchmark.pilot.embeddings.x_rank_candidates__mutmut_24": "c8115882ee47b0dd948a062bfe900b126accd40a75a0b8a7088afc2b73e14eae",
    "georeset_text_label_benchmark.pilot.embeddings.x_rank_candidates__mutmut_25": "7c4dbf709f985eae38b186de9470ef801d000254a365c07acace60f22e291253",
    "georeset_text_label_benchmark.pilot.metrics.x__validate_predictions__mutmut_27": "27412a2a5a58e5b8360085f04ce422598082db5be9c3a4491085fbc008b87451",
    "georeset_text_label_benchmark.pilot.metrics.x__validate_predictions__mutmut_30": "8c1e2b3d1d92180e3cdbf00e0ae977ba3d4ff1c6cfd3bdbd0aaea4e0156a1825",
    "georeset_text_label_benchmark.pilot.metrics.x__validate_predictions__mutmut_31": "902d980ecea4277c5fe3b2cebac2ac0aeba48b2c2ea0bc13c9f9f3a1cc130e21",
    "georeset_text_label_benchmark.pilot.metrics.x_class_breakdown__mutmut_20": "410b387cb6b02b0cf0d6a1c005d90c6ccbb7312323bda8563a361dfc5f770163",
    "georeset_text_label_benchmark.pilot.metrics.x_class_breakdown__mutmut_23": "a1892ea2eb85b73f56df888faa7310920e22a549ab39ea8fc086f4eb48800cf0",
    "georeset_text_label_benchmark.pilot.metrics.x_class_breakdown__mutmut_24": "d6c17c26d6cdf434e898792f80569e595e699be4f9eb67270c71adbaa85afccd",
    "georeset_text_label_benchmark.pilot.metrics.x_single_label_report__mutmut_15": "e23a83cfe83838be7ad8ad2ef2125ea305504e8808f709294d240003180999f6",
    "georeset_text_label_benchmark.pilot.metrics.x_single_label_report__mutmut_18": "22083d814d4a8b41f72a21a7d1f7a4679a990335b5b36a86100222e5c0084f84",
    "georeset_text_label_benchmark.pilot.metrics.x_single_label_report__mutmut_19": "e654d8b6d1e7369f736b5dbfac4d2bb12cae203dfd7016fa2558dad0d28fd8be",
    "georeset_text_label_benchmark.pilot.publication.x__probe_exclusive_link__mutmut_15": "3daacc58a796a1ba62bfad571483a41842dc8c70f98f84c4209e5de45fd7f5eb",
    "georeset_text_label_benchmark.pilot.publication.x__probe_exclusive_link__mutmut_16": "7839001b3df29a1f19bdca975cbfbf3aa20916aa3bfa49bc4201e92a5a7c627d",
    "georeset_text_label_benchmark.pilot.publication.x__probe_exclusive_link__mutmut_18": "7b578c33dd84612079f8f3ccfb82834d780d0cc8e472b555bf9ffdaba5e7e30f",
    "georeset_text_label_benchmark.pilot.publication.x__probe_exclusive_link__mutmut_19": "f096eebee00b03f00f4505557510d6525b850739b9e76591755538b7468d437c",
    "georeset_text_label_benchmark.pilot.publication.x_ensure_publication_supported__mutmut_14": "363c78a6a00872d3a6ebe3405741286bbb63551da5dcb7b45b828d61010d1b71",
    "georeset_text_label_benchmark.pilot.publication.x_ensure_publication_supported__mutmut_15": "3e361f1804d81a434dc3b63be842dce8a73f730cd11b053b7a84654d86ddb1b3",
    "georeset_text_label_benchmark.pilot.publication.x_ensure_publication_supported__mutmut_18": "b09ffd1ff19b25363415a9a83ab1dcb9badf20efe9db252b3ce7c2c95f3fc11f",
    "georeset_text_label_benchmark.pilot.publication.x_ensure_publication_supported__mutmut_19": "35d3016d27ce2b4357c57d695ec301a4fde55f223730d1d2b3702c66fe02ef4d",
    "georeset_text_label_benchmark.pilot.publication.x_ensure_publication_supported__mutmut_21": "affb4ef23b4bb5bc65a08c5349e9c7c518d18346ea56b85cf9d747edb37cca77",
    "georeset_text_label_benchmark.pilot.publication.x_ensure_publication_supported__mutmut_22": "42b5aa3f02008be8a5606d42fb4e21542ae86357483e4147492839b768c9fff0",
    "georeset_text_label_benchmark.pilot.publication.x_ensure_publication_supported__mutmut_4": "e72d89b0dc76228ab868747e7a0f78b9c709fba56f31f6c7a08804af6181215a",
    "georeset_text_label_benchmark.pilot.publication.x_ensure_publication_supported__mutmut_6": "0bcff19cb8814b766822b1a203e47b0ef0bd9fbe93f15b2e7626565ca97b2555",
    "georeset_text_label_benchmark.pilot.publication.x_ensure_publication_supported__mutmut_8": "7cce0b20e593e3e1d93c71ba9f8ce7952aeddfd8f1a4e3463188e9e482bafe5f",
    "georeset_text_label_benchmark.pilot.publication.x_ensure_publication_supported__mutmut_9": "15be2406b3fad0c742afc5504395358c0f75d185c269912c52f13872fb50ca9c",
    "georeset_text_label_benchmark.pilot.runner.x__read_frozen_sample__mutmut_7": "f1b4b555f5be6ce35f591c3723b437c6063c7abbd13b253c2a93f18810691de7",
    "georeset_text_label_benchmark.pilot.runner.x__runtime_metadata__mutmut_10": "66c043a5008b2baab6131e970b88f4650ea652d440f554f7048776d461232e7d",
    "georeset_text_label_benchmark.pilot.runner.x__runtime_metadata__mutmut_15": "6d3c11b3caff7245cacd86adb3669bbd0787cc5d3ced5281eb07fe38eae1e125",
    "georeset_text_label_benchmark.pilot.runner.x__runtime_metadata__mutmut_20": "32e8ab89ece1a164e527c3fc80fc1361b26cc07965a89f0e488a496694fe47b2",
    "georeset_text_label_benchmark.pilot.runner.x__sha256_json__mutmut_17": "36609ae3fb0b453e5906bd39af5b9067b5b8ebdc31e60279b30eaf0d73ac7a92",
    "georeset_text_label_benchmark.pilot.runner.x__sha256_json__mutmut_3": "618529fcc944e1e67ec0102c604da5bd6eb93bb0f0f8919f715f41b260a31675",
    "georeset_text_label_benchmark.pilot.runner.x__write_json_exclusive__mutmut_14": "2803739de9f43ade2ae855dcf19d4a41dda51a6c9453294b5e41e6523153851a",
    "georeset_text_label_benchmark.pilot.runner.x__write_outputs__mutmut_14": "a99992fd7823bf9e66b06c7603abc43cc49594b740724c72bc1451e34c881adf",
    "georeset_text_label_benchmark.pilot.runner.x_freeze_sample__mutmut_32": "31eb00746606888d651f6e19a560c04f439404473045ab02d1c23803f811a8fb",
    "georeset_text_label_benchmark.pilot.runner.x_freeze_sample__mutmut_34": "f592f802595ace7e53bd468682dc98ba83a06d70700bd6932106884570f36dcf",
    "georeset_text_label_benchmark.pilot.runner.x_run_pilot__mutmut_98": "bdbe1bd04e8b0974f0d4e3dbd60cda07e15d3e72104d1d48193373cf424350fa",
    "georeset_text_label_benchmark.pilot.runner.x_sha256_file__mutmut_9": "3678fd7c4955e8b29dba31c6185d2bb948418be50482141a9019fedf93c407de",
    "georeset_text_label_benchmark.pilot.sampling.x__rank_rows__mutmut_5": "a0d87273a7f12e7e5b9e47105fd97be3413da5d8c7c5467c37cd9db17d00bb03",
    "georeset_text_label_benchmark.pilot.sampling.x__sample_id__mutmut_11": "15828e13b1733bacc4e8aa688123d4e388339035c82008162bcd41b4dd7d993f",
    "georeset_text_label_benchmark.pilot.sampling.x__sample_id__mutmut_15": "5590a66e243ab9ee9a7b3cc68a1a049aef203b8d688c9f5c72e55376fd354b74",
    "georeset_text_label_benchmark.pilot.sampling.x__sample_id__mutmut_4": "036e478e6eda1562c04e91c457bba2d6ad3450857b201f044061ef52761f64be",
    "georeset_text_label_benchmark.pilot.sampling.x__validate_sentence_hash__mutmut_5": "93966eb39dc54e4d89032ccbf73e97f7181a1fcd05085e63a2b85e6a5c170a39",
    "georeset_text_label_benchmark.pipeline.x__build_run__mutmut_21": "232cbfe1f08997724580af277de72347433894da72c82a47008ca4c24537d435",
    "georeset_text_label_benchmark.quality.mutation.x__mutation_fingerprint__mutmut_20": "db9169a83216ba1277b993ce9540f311abe9026c6756032073b3c82ecf8a743c",
    "georeset_text_label_benchmark.pilot.cli.x_main__mutmut_13": "9cf9a4c27094fcb51453ad8ecd02cfbc09a32ee7c9b6fed43bd155892ecc4eb0",
    "georeset_text_label_benchmark.pilot.metrics.x__true_positives__mutmut_4": "e79eeed54d1dbac09488bcf4bc198118b11b5746c4570e8a65c9d8fa8eb94da5",
    "georeset_text_label_benchmark.pilot.metrics.x__true_positives__mutmut_7": "825f4587b2e13db3bbc83d24ec9944ef136a29e90d3b008d3a2984b34186c99d",
    "georeset_text_label_benchmark.pilot.metrics.x__true_positives__mutmut_8": "9e98b84cc1aa08ca5dd153aeb44dca92b0e71f75ccae8d58415c473a141b3f75",
}

REVIEWED_EXEMPTIONS = {
    name: MutationExemption(EQUIVALENT_FINGERPRINTS.get(name, ""), rationale)
    for name, rationale in EQUIVALENT_RATIONALES.items()
}
EQUIVALENT_MUTANTS = REVIEWED_EXEMPTIONS


def _result_line(line: str) -> tuple[str, str]:
    match = RESULT_PATTERN.fullmatch(line)
    if match is None:
        raise ValueError(f"unrecognized mutmut result line: {line!r}")
    return match.group(1), match.group(2)


def _parse_results(output: str) -> dict[str, str]:
    records = [_result_line(line) for line in output.splitlines() if line.strip()]
    results = dict(records)
    if not records:
        raise ValueError("mutmut returned no mutation results")
    if len(results) != len(records):
        raise ValueError("mutmut returned duplicate mutant identifiers")
    return results


def _mutation_fingerprint(output: str, mutant_name: str) -> str:
    """Hash the reviewed unified patch, including its file and hunk locations."""
    lines = output.splitlines()
    _validate_mutation_header(lines, mutant_name)

    canonical = lines[1:]
    if any(not _valid_mutation_diff_line(line) for line in canonical):
        raise ValueError(f"mutmut show returned an invalid diff for {mutant_name}")
    if not _has_complete_mutation_diff(canonical):
        raise ValueError(f"mutmut show returned an incomplete diff for {mutant_name}")
    return hashlib.sha256("\n".join(canonical).encode("utf-8")).hexdigest()


def _validate_mutation_header(lines: list[str], mutant_name: str) -> None:
    if not lines or lines[0] != f"# {mutant_name}: survived":
        raise ValueError(f"mutmut show returned an unexpected header for {mutant_name}")


def _valid_mutation_diff_line(line: str) -> bool:
    if line.startswith(("---", "+++")):
        return line.startswith(("--- ", "+++ "))
    return line.startswith(("@@", "+", "-", " "))


def _has_complete_mutation_diff(lines: list[str]) -> bool:
    return _has_diff_paths(lines) and _has_changed_diff_content(lines)


def _has_diff_paths(lines: list[str]) -> bool:
    headers = {line[:4] for line in lines if line.startswith(("--- ", "+++ "))}
    return headers == {"--- ", "+++ "}


def _has_changed_diff_content(lines: list[str]) -> bool:
    return any(
        line.startswith(("+", "-")) and not line.startswith(("--- ", "+++ ")) for line in lines
    )


def _matches_reviewed_fingerprint(name: str, fingerprint: str | None) -> bool:
    exemption = REVIEWED_EXEMPTIONS.get(name)
    return (
        exemption is not None
        and len(exemption.fingerprint) == 64
        and fingerprint == exemption.fingerprint
    )


def _failures(results: dict[str, str], fingerprints: Mapping[str, str]) -> list[str]:
    return [
        f"{name}: {status}"
        for name, status in sorted(results.items())
        if status != "killed"
        and not (
            status == "survived" and _matches_reviewed_fingerprint(name, fingerprints.get(name))
        )
    ]


def _killed_count(results: dict[str, str]) -> int:
    return sum(status == "killed" for status in results.values())


def _equivalent_survivors(results: dict[str, str], fingerprints: Mapping[str, str]) -> list[str]:
    return sorted(
        name
        for name, status in results.items()
        if status == "survived" and _matches_reviewed_fingerprint(name, fingerprints.get(name))
    )


def _read_results() -> dict[str, str] | None:
    completed = subprocess.run(
        ["mutmut", "results", "--all=true"], capture_output=True, text=True, check=False
    )
    if completed.returncode:
        print(completed.stderr or completed.stdout)
        return None
    try:
        results = _parse_results(completed.stdout)
    except ValueError as error:
        print(str(error))
        return None
    return results


def _survivor_names(results: Mapping[str, str]) -> list[str]:
    return [name for name, status in sorted(results.items()) if status == "survived"]


def _read_mutation_patch(name: str) -> str:
    completed = subprocess.run(
        ["mutmut", "show", name], capture_output=True, text=True, check=False
    )
    if completed.returncode:
        raise ValueError(f"mutmut show failed for {name}: {completed.stderr or completed.stdout}")
    return completed.stdout


def _read_mutation_fingerprints(results: Mapping[str, str]) -> dict[str, str]:
    fingerprints = {}
    for name in _survivor_names(results):
        patch = _read_mutation_patch(name)
        fingerprint = _mutation_fingerprint(patch, name)
        fingerprints[name] = fingerprint
        if not _matches_reviewed_fingerprint(name, fingerprint):
            print(f"Unresolved mutation patch for {name}:")
            print(patch, end="" if patch.endswith("\n") else "\n")
            print(f"Mutation fingerprint: {fingerprint}")
    return fingerprints


def _report_results(
    results: dict[str, str],
    fingerprints: Mapping[str, str],
    patches: Mapping[str, str] | None = None,
) -> int:
    failures = _failures(results, fingerprints)
    print(f"Mutation results: {_killed_count(results)}/{len(results)} killed")
    for name in _equivalent_survivors(results, fingerprints):
        exemption = REVIEWED_EXEMPTIONS[name]
        print(
            f"Equivalent mutant documented: {name}: {exemption.rationale} "
            f"(diff sha256: {exemption.fingerprint})"
        )
    if failures:
        print("Unresolved mutation results:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    return 0


def main() -> int:
    results = _read_results()
    if results is None:
        return 1
    try:
        fingerprints = _read_mutation_fingerprints(results)
    except ValueError as error:
        print(str(error))
        return 1
    return _report_results(results, fingerprints)
