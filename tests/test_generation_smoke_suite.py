import numpy as np
import torch

import inference
import utils
from models import VAE


class _FakeLatentDistribution:
    def __init__(self, x):
        self.mean = x * 0.1
        self.std = torch.ones_like(x) * 0.05


class _FakeDecodeResult:
    def __init__(self, sample):
        self.sample = sample


class _FakeVAE:
    def __init__(self, scaling_factor):
        self.scaling_factor = scaling_factor
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def encode(self, x):
        return _FakeLatentDistribution(x)

    def decode(self, latents):
        decoded = torch.sin((latents[:, :3, :, :] - 0.5) * 4.0)
        return _FakeDecodeResult(decoded)


class _FakeScheduler:
    def get_schedule(self, steps):
        return torch.linspace(1.0, 0.1, steps)

    def get_truncated_schedule(self, steps, scheduled_steps):
        return self.get_schedule(max(steps, 1))


class _FakeSampler:
    def __init__(self):
        self.scheduler = _FakeScheduler()

    def prepare_noise(self, noise, schedule):
        return noise + schedule[0] * 0.1

    def step(self, latents, schedule, i, noise_fn):
        return latents * 0.85 + noise_fn() * 0.15 + (i + 1) * 0.01

    def prepare_latents(self, latents, noise, schedule):
        return latents * 0.9 + noise * 0.1


class _FakeDenoiser:
    def __init__(self):
        self.predictions = []
        self.unet = type("U", (), {"dtype": torch.float32})()

    def set_step(self, step):
        self.predictions.append(step)


def _noise():
    base = torch.ones((1, 4, 8, 8), dtype=torch.float32) * 0.25
    x = torch.linspace(0.0, 1.0, 8, dtype=torch.float32).view(1, 1, 1, 8)
    y = torch.linspace(0.0, 1.0, 8, dtype=torch.float32).view(1, 1, 8, 1)
    return base + x * 0.05 + y * 0.05


def _assert_non_black(images):
    image = np.asarray(images[0], dtype=np.float32)
    assert image.mean() > 0.0
    assert image.max() > image.min()


def _run_txt2img_and_img2img_smoke(model_type):
    vae = _FakeVAE(VAE.get_config(model_type)["scaling_factor"])
    sampler = _FakeSampler()
    denoiser = _FakeDenoiser()

    txt2img_latents = inference.txt2img(denoiser, sampler, _noise, steps=3, callback=lambda *_: None)
    txt2img_images = utils.decode_images(vae, txt2img_latents)
    _assert_non_black(txt2img_images)

    img2img_input = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
    img2img_latents = inference.img2img(
        img2img_input,
        denoiser,
        sampler,
        _noise,
        steps=4,
        do_exact_steps=False,
        strength=0.75,
        callback=lambda *_: None,
    )
    img2img_images = utils.decode_images(vae, img2img_latents)
    _assert_non_black(img2img_images)


def test_smoke_sd15_txt2img_and_img2img_non_black_outputs():
    _run_txt2img_and_img2img_smoke("SDv1")


def test_smoke_sd2_txt2img_and_img2img_non_black_outputs():
    _run_txt2img_and_img2img_smoke("SDv2")


def test_smoke_sdxl_txt2img_and_img2img_non_black_outputs():
    _run_txt2img_and_img2img_smoke("SDXL-Base")
