import os
from typing import Union

from peft import LoraConfig, get_peft_model, PeftModel
from safetensors.torch import load_file

def has_lora(transformer, adapter_name="default"):
    """Check if transformer already has LoRA adapter applied."""
    return isinstance(transformer, PeftModel) and adapter_name in getattr(transformer, 'peft_config', {})


def add_lora(transformer, lora_rank, init_lora_weights: Union[bool, str] = "gaussian"):
    """
    Add LoRA layers to a transformer model.

    Args:
        transformer: The transformer model to add LoRA to
        lora_rank: Rank of the LoRA layers
        init_lora_weights: LoRA initialization method
            - "gaussian" (default, preserves pre-existing behavior for every caller that
              doesn't pass this): both A and B are Gaussian random
            - True: Standard init, A=Kaiming, B=zeros (required for zero LoRA trick) --
              pass explicitly (fibo_edit_next's init_handler.py does)
    """
    target_modules = [
        # HF Lora Layers
        "attn.to_k",
        "attn.to_q",
        "attn.to_v",
        "attn.to_out.0",
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "ff.net.0.proj",
        "ff.net.2",
        "ff_context.net.0.proj",
        "ff_context.net.2",
        "proj_mlp",
        # +  layers that exist on ostris ai-toolkit / replicate trainer
        "norm1_context.linear",
        "norm1.linear",
        "norm.linear",
        "proj_out",
    ]
    transformer_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        init_lora_weights=init_lora_weights,
        target_modules=target_modules,
    )
    transformer = get_peft_model(transformer, transformer_lora_config, adapter_name="default")
    return transformer


def load_lora(transformer, input_dir, is_trainable=False):

    """Load LoRA weights into an existing PeftModel.
    Assumes add_lora has already been called to create the LoRA architecture.
    Handles Diffusers format (transformer.*.lora_A/B.weight) by converting to PEFT format.
    """

    assert has_lora(transformer, adapter_name="default")

    # Try different weight file names
    weights_file = os.path.join(input_dir, "pytorch_lora_weights.safetensors")
    if not os.path.isfile(weights_file):
        weights_file = os.path.join(input_dir, "adapter_model.safetensors")
    if not os.path.isfile(weights_file):
        raise FileNotFoundError(f"No LoRA weights file found in {input_dir}")
    
    state_dict = load_file(weights_file)
    
    # Convert Diffusers format to PEFT format if needed
    # Diffusers: transformer.layer.lora_A.weight -> PEFT: base_model.model.layer.lora_A.default.weight
    converted_state_dict = {}
    for key, value in state_dict.items():

        new_key = key

        # Remove 'transformer.' prefix if present
        if new_key.startswith("transformer."):
            new_key = new_key[len("transformer."):]
        # remove base_model.model prefix if present since we add it anyway
        if new_key.startswith("base_model.model."):
            new_key = new_key[len("base_model.model."):]
        
        # Convert lora_A.weight -> lora_A.default.weight (PEFT format)
        if ".lora_A.weight" in new_key:
            new_key = new_key.replace(".lora_A.weight", ".lora_A.default.weight")
        elif ".lora_B.weight" in new_key:
            new_key = new_key.replace(".lora_B.weight", ".lora_B.default.weight")
        
        # Add base_model.model prefix for PEFT
        new_key = "base_model.model." + new_key
        converted_state_dict[new_key] = value
    
    missing, unexpected = transformer.load_state_dict(converted_state_dict, strict=False)
    assert len(unexpected) == 0, f"Unexpected keys when loading LoRA: {unexpected}"
    if not is_trainable:
        for param in transformer.parameters():
            param.requires_grad = False
    
    return transformer
