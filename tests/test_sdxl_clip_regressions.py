import types

import torch

from clip import CustomCLIP, CustomSDXLCLIP
from models import CLIP


class _FakeTextModel:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def __call__(self, _input_ids):
        hidden = torch.ones((2, 77, self.hidden_size), dtype=torch.float32)
        pooled = torch.arange(2 * self.hidden_size, dtype=torch.float32).reshape(2, self.hidden_size)
        return types.SimpleNamespace(hidden_states=[hidden, hidden], pooler_output=pooled)

    def final_layer_norm(self, x):
        return x


def test_custom_clip_with_projection_keeps_batch_pooled_embeddings():
    clip = CustomCLIP.__new__(CustomCLIP)
    torch.nn.Module.__init__(clip)
    clip.text_model = _FakeTextModel(4)
    clip.text_projection = torch.nn.Linear(4, 3, bias=False)

    cond, emb = clip.forward([1, 2, 3], clip_skip=1)

    assert cond.shape == (2, 77, 4)
    assert emb.shape == (2, 3)


def test_custom_sdxl_clip_keeps_batch_pooled_embeddings():
    sdxl = CustomSDXLCLIP.__new__(CustomSDXLCLIP)
    torch.nn.Module.__init__(sdxl)

    open_clip = types.SimpleNamespace(
        text_model=_FakeTextModel(1280),
        text_projection=torch.nn.Linear(1280, 1280, bias=False),
    )
    ldm_clip = types.SimpleNamespace(text_model=_FakeTextModel(768))

    sdxl.open_clip = open_clip
    sdxl.ldm_clip = ldm_clip

    cond, emb = sdxl.forward([1, 2, 3], clip_skip=1)

    assert cond.shape == (2, 77, 2048)
    assert emb.shape == (2, 1280)


def test_refiner_textual_inversion_uses_openclip_slice_only():
    clip = CLIP.__new__(CLIP)
    clip.model_type = "SDXL-Base"
    clip.model_variant = "Refiner"
    clip.tokenizer = lambda text: {"input_ids": [49406, 100, 101, 49407]}

    vec = torch.randn(4, 2048)
    clip.set_textual_inversions({"my_embedding": vec})

    _, stored = clip.textual_inversions[0]
    assert stored.shape == (4, 1280)
