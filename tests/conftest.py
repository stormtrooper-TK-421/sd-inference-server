import sys
import types


if "lycoris" not in sys.modules:
    module = types.ModuleType("lycoris")
    module.logger = types.SimpleNamespace(disabled=False)

    class _Base:
        def forward(self, x, *args, **kwargs):
            return x

    module.LoConModule = _Base
    module.LohaModule = _Base
    module.LokrModule = _Base
    module.FullModule = _Base
    module.NormModule = _Base
    module.kohya = types.SimpleNamespace(
        LycorisNetworkKohya=object,
        create_network_from_weights=lambda *args, **kwargs: (None, None),
    )
    sys.modules["lycoris"] = module

try:
    import diffusers  # noqa: F401
except Exception:
    diffusers_module = types.ModuleType("diffusers")

    class _BaseModel:
        def __init__(self, *args, **kwargs):
            pass

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def to(self, *args, **kwargs):
            return self

        def state_dict(self):
            return {}

    diffusers_module.AutoencoderKL = _BaseModel
    diffusers_module.UNet2DConditionModel = _BaseModel

    models_module = types.ModuleType("diffusers.models")
    controlnet_module = types.ModuleType("diffusers.models.controlnet")
    controlnet_module.ControlNetModel = _BaseModel

    autoencoders_module = types.ModuleType("diffusers.models.autoencoders")
    vae_module = types.ModuleType("diffusers.models.autoencoders.vae")

    class _DiagonalGaussianDistribution:
        def __init__(self, moments):
            self.mean = moments
            self.std = moments

    vae_module.DiagonalGaussianDistribution = _DiagonalGaussianDistribution

    lora_module = types.ModuleType("diffusers.models.lora")
    lora_module.LoRACompatibleConv = object
    lora_module.LoRACompatibleLinear = object

    models_module.lora = lora_module
    models_module.controlnet = controlnet_module
    models_module.autoencoders = autoencoders_module
    diffusers_module.models = models_module

    sys.modules["diffusers"] = diffusers_module
    sys.modules["diffusers.models"] = models_module
    sys.modules["diffusers.models.controlnet"] = controlnet_module
    sys.modules["diffusers.models.autoencoders"] = autoencoders_module
    sys.modules["diffusers.models.autoencoders.vae"] = vae_module
    sys.modules["diffusers.models.lora"] = lora_module

try:
    import transformers  # noqa: F401
except Exception:
    transformers_module = types.ModuleType("transformers")
    transformers_module.CLIPTextConfig = object
    transformers_module.CLIPTokenizer = object
    transformers_module.CLIPTextModel = object

    tf_models_module = types.ModuleType("transformers.models")
    tf_clip_module = types.ModuleType("transformers.models.clip")
    tf_clip_modeling_module = types.ModuleType("transformers.models.clip.modeling_clip")

    class _CLIPTextTransformer:
        def __init__(self, *args, **kwargs):
            self.config = types.SimpleNamespace(output_attentions=False, output_hidden_states=False, use_return_dict=True)

    def _create_4d_causal_attention_mask(*args, **kwargs):
        return None

    class _BaseModelOutputWithPooling:
        def __init__(self, *args, **kwargs):
            self.last_hidden_state = None
            self.pooler_output = None
            self.hidden_states = None
            self.attentions = None

    tf_clip_modeling_module.CLIPTextTransformer = _CLIPTextTransformer
    tf_clip_modeling_module._create_4d_causal_attention_mask = _create_4d_causal_attention_mask
    tf_clip_modeling_module.BaseModelOutputWithPooling = _BaseModelOutputWithPooling

    sys.modules["transformers"] = transformers_module
    sys.modules["transformers.models"] = tf_models_module
    sys.modules["transformers.models.clip"] = tf_clip_module
    sys.modules["transformers.models.clip.modeling_clip"] = tf_clip_modeling_module

if "ultralytics" not in sys.modules:
    ultralytics_module = types.ModuleType("ultralytics")

    class _YOLO:
        def __init__(self, *args, **kwargs):
            pass

    ultralytics_module.YOLO = _YOLO
    ultralytics_utils_module = types.ModuleType("ultralytics.utils")
    ultralytics_utils_module.LOGGER = types.SimpleNamespace(disabled=False)

    sys.modules["ultralytics"] = ultralytics_module
    sys.modules["ultralytics.utils"] = ultralytics_utils_module
