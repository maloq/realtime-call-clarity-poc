from __future__ import annotations


def wer_cer(reference: str, hypothesis: str) -> dict[str, float | None]:
    try:
        import jiwer
    except Exception:
        return {"wer": None, "cer": None}
    return {"wer": float(jiwer.wer(reference, hypothesis)), "cer": float(jiwer.cer(reference, hypothesis))}
