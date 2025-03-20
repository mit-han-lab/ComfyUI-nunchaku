import logging
import os

import folder_paths
from nunchaku.lora.flux import is_nunchaku_format, to_nunchaku

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NunchakuFluxLoraLoader")


class NunchakuFluxLoraLoader:
    def __init__(self):
        self.cur_lora_name = "None"

    @classmethod
    def INPUT_TYPES(s):
        lora_name_list = ["None", *folder_paths.get_filename_list("loras")]

        prefixes = folder_paths.folder_names_and_paths["diffusion_models"][0]
        base_model_paths = set()
        for prefix in prefixes:
            if os.path.exists(prefix) and os.path.isdir(prefix):
                base_model_paths_ = os.listdir(prefix)
                base_model_paths_ = [
                    folder
                    for folder in base_model_paths_
                    if not folder.startswith(".") and os.path.isdir(os.path.join(prefix, folder))
                ]
                base_model_paths.update(base_model_paths_)
        base_model_paths = sorted(list(base_model_paths))

        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "lora_name": (lora_name_list, {"tooltip": "The name of the LoRA."}),
                "base_model_name": (
                    base_model_paths,
                    {
                        "tooltip": "If the lora format is SVDQuant, this field has no use. Otherwise, the base model's state dictionary is required for converting the LoRA weights to SVDQuant."
                    },
                ),
                "lora_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "How strongly to modify the diffusion model. This value can be negative.",
                    },
                ),
                "save_converted_lora": (
                    ["disable", "enable"],
                    {
                        "tooltip": "If enabled, the converted LoRA will be saved as a .safetensors file in the save directory of your LoRA file."
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    OUTPUT_TOOLTIPS = ("The modified diffusion model.",)
    FUNCTION = "load_lora"
    TITLE = "Nunchaku FLUX.1 LoRA Loader"

    CATEGORY = "Nunchaku"
    DESCRIPTION = (
        "LoRAs are used to modify the diffusion model, "
        "altering the way in which latents are denoised such as applying styles. "
        "Currently, only one LoRA nodes can be applied."
    )

    def load_lora(
        self,
        model,
        lora_name: str,
        base_model_name: str,
        lora_strength: float,
        save_converted_lora: str,
    ):
        if self.cur_lora_name == lora_name:
            if self.cur_lora_name == "None":
                pass  # Do nothing since the lora is None
            else:
                model.model.diffusion_model.model.set_lora_strength(lora_strength)
        else:
            if lora_name == "None":
                model.model.diffusion_model.model.set_lora_strength(0)
            else:
                try:
                    lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
                except FileNotFoundError:
                    lora_path = lora_name
                if not is_nunchaku_format(lora_path):
                    prefix = os.path.join(folder_paths.models_dir, "diffusion_models")
                    base_model_path = os.path.join(prefix, base_model_name, "transformer_blocks.safetensors")
                    if not os.path.exists(base_model_path):
                        # download from huggingface
                        base_model_path = os.path.join(base_model_name, "transformer_blocks.safetensors")

                    output_path = None
                    if save_converted_lora == "enable":
                        dirname = os.path.dirname(lora_path)
                        basename = os.path.basename(lora_path)
                        precision = "fp4" if "fp4" in base_model_name else "int4"
                        converted_name = f"svdq-{precision}-{basename}"
                        output_path = os.path.join(dirname, converted_name)

                    state_dict = to_nunchaku(lora_path, base_model_path, output_path=output_path)
                    model.model.diffusion_model.model.update_lora_params(state_dict)
                else:
                    model.model.diffusion_model.model.update_lora_params(lora_path)
                model.model.diffusion_model.model.set_lora_strength(lora_strength)
            self.cur_lora_name = lora_name

        return (model,)

    # TODO: add lora to the text encoder
