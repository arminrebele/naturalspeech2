"""Message-scoped suppression of known-benign torch / third-party log spam.

Every filter targets a SPECIFIC message or logger → unknown/new warnings stay visible. Call once at
the entry of each process that drives torch (train, eval_daemon, inference CLI). Each item below is
cosmetic; the per-item rationale lives in the log-analysis notes.

Two distinct mechanisms (don't conflate them):
  - warnings.warn(...)  → silence with warnings.filterwarnings("ignore", message=...). Suppressed at
    warn() time, so logging.captureWarnings(True) never sees them.
  - logger.warning(...) → a real logging record; needs a logging.Filter on the emitting logger.
"""
import logging
import warnings


class _MaybeGuardRelFilter(logging.Filter):
    """Drop symbolic_shapes '_maybe_guard_rel() ... non-relation' records: dynamo skipping
    symbol-range refinement on an OR-guard (broadcast / STFT as_strided layout) at (re)compile. The
    guard is still installed → zero correctness signal; recompile churn is covered by
    Compile/unique_graphs telemetry instead."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "_maybe_guard_rel() was called on non-relation" not in record.getMessage()


class _WordsMismatchFilter(logging.Filter):
    """Drop phonemizer's 'words count mismatch on N% of the lines' records. espeak's word-count
    heuristic logs this per phonemized utterance even in words_mismatch='ignore' mode (the Ignore
    processor still calls _resume(), which logs the summary), so eval generation floods the daemon log
    (~80 lines/eval). A Filter, NOT setLevel(ERROR): phonemizer's get_logger() resets the 'phonemizer'
    logger level back to WARNING when the espeak backend is built (after our install) but never clears
    filters, and its NullHandler only blocks phonemizer's OWN stderr — the record still propagates to
    the root file handler. Targets the one message → real phonemizer warnings/errors stay visible."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "words count mismatch" not in record.getMessage()


def install_warning_filters() -> None:
    """Register all suppressions for the current process (idempotent enough — called once per entry)."""

    # --- torch logging records (logging.Logger, NOT warnings.warn) ---
    logging.getLogger("torch.fx.experimental.symbolic_shapes").addFilter(_MaybeGuardRelFilter())

    # --- torch python warnings (warnings.warn), matched on the leading message text ---
    # Inductor falls back to eager/aten for the mel-path STFT complex ops (perf-only, tiny op vs the
    # 40-block denoiser) — can't fix without dropping compile on mel gen, not worth it.
    warnings.filterwarnings(
        "ignore", message=r"Torchinductor does not support code generation for complex operators")
    # TF32 hint: matmul precision deliberately kept at 'highest' to protect the aligner's FP32 GEMM
    # and Encodec's FP32 codebook-distance island — enabling TF32 would perturb exactly those. Silence.
    warnings.filterwarnings(
        "ignore",
        message=r"TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled")

    # --- phonemizer: espeak word-count heuristic on our punctuation-split fragments (benign — mismatches
    # are inherent to fragment-wise phonemization). Message-scoped Filter, not setLevel: phonemizer's
    # get_logger() resets the logger level at backend build (clobbering setLevel) but leaves filters. ---
    logging.getLogger("phonemizer").addFilter(_WordsMismatchFilter())

    # --- third-party s3prl/WavLM deprecations (in-process eval load path only), benign ---
    warnings.filterwarnings("ignore", message=r"torch\.nn\.utils\.weight_norm is deprecated")
    warnings.filterwarnings(
        "ignore", message=r"Support for mismatched key_padding_mask and attn_mask is deprecated")
