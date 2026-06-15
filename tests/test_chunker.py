import torch

from callclarity.streaming.chunker import iter_audio_chunks, reconstruct_chunks


def test_chunker_preserves_sample_count():
    waveform = torch.randn(1, 1607)
    chunks = list(iter_audio_chunks(waveform, 16000, chunk_ms=10))
    reconstructed = reconstruct_chunks(chunks)
    assert reconstructed.shape[-1] == waveform.shape[-1]
    assert torch.allclose(reconstructed, waveform)
