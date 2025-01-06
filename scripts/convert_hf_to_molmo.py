import argparse
import logging
import math
import os
from pathlib import Path
from typing import Dict, Any

import torch
import einops
from flax.traverse_util import flatten_dict, unflatten_dict
from transformers import AutoModel, AutoModelForCausalLM, CLIPModel, SiglipModel

from launch_scripts.utils import VISION_BACKBONES, LLMS, DEFAULT_LOAD_PATHS
from olmo import VisionBackboneConfig, ModelConfig, Molmo, BlockType
from olmo.util import prepare_cli_environment


def interpolate_position_embeddings(
    position_embeddings: torch.Tensor,
    num_patches: int,
    dim: int,
    patch_size: int,
    height: int,
    width: int,
    num_prefix_tokens: int = 1,
) -> torch.Tensor:
    from torch import nn as torch_nn

    num_positions = position_embeddings.shape[1] - num_prefix_tokens
    if num_patches == num_positions and height == width:
        return position_embeddings
    class_pos_embed = position_embeddings[:, :num_prefix_tokens]
    patch_pos_embed = position_embeddings[:, num_prefix_tokens:]
    height = height // patch_size
    width = width // patch_size
    # we add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    height, width = height + 0.1, width + 0.1
    patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
    patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
    target_dtype = patch_pos_embed.dtype
    patch_pos_embed = torch_nn.functional.interpolate(
        patch_pos_embed.to(dtype=torch.float32),
        scale_factor=(float(height / math.sqrt(num_positions)), float(width / math.sqrt(num_positions))),
        mode="bicubic",
        align_corners=False,
    ).to(dtype=target_dtype)
    if int(height) != patch_pos_embed.shape[-2] or int(width) != patch_pos_embed.shape[-1]:
        raise ValueError("Width or height does not match with the interpolated position embeddings")
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed, patch_pos_embed), dim=1)


def convert_state_dict_clip(state_dict, vision_config: VisionBackboneConfig) -> Dict[str, Any]:
    state_dict = unflatten_dict(state_dict, sep=".")

    resblocks = {}
    for layer in range(vision_config.image_num_layers):
        layer_dict = state_dict["encoder"]["layers"][str(layer)]
        q, k, v, o = [
            layer_dict["self_attn"][f"{x}_proj"].pop("weight")
            for x in ["q", "k", "v", "out"]
        ]
        q_b, k_b, v_b, o_b = [
            layer_dict["self_attn"][f"{x}_proj"].pop("bias")
            for x in ["q", "k", "v", "out"]
        ]

        w1, w2 = [layer_dict["mlp"][f"{x}"].pop("weight") for x in ["fc1", "fc2"]]
        w1_b, w2_b = [layer_dict["mlp"][f"{x}"].pop("bias") for x in ["fc1", "fc2"]]

        mapped_layer_dict = {
            "attention": {
                "wq": dict(weight=q, bias=q_b),
                "wk": dict(weight=k, bias=k_b),
                "wv": dict(weight=v, bias=v_b),
                "wo": dict(weight=o, bias=o_b),
            },
            "feed_forward": {
                "w1": dict(weight=w1, bias=w1_b),
                "w2": dict(weight=w2, bias=w2_b), 
            },
            "attention_norm": {
                "weight": layer_dict["layer_norm1"].pop("weight"),
                "bias": layer_dict["layer_norm1"].pop("bias"),
            },
            "ffn_norm": {
                "weight": layer_dict["layer_norm2"].pop("weight"),
                "bias": layer_dict["layer_norm2"].pop("bias"),
            }
        }

        resblocks[str(layer)] = mapped_layer_dict
    
    # We accidentally set the number of layers for OpenAI CLIP ViT to 23 in experiments
    if str(vision_config.image_num_layers) in state_dict["encoder"]["layers"]:
        del state_dict["encoder"]["layers"][str(vision_config.image_num_layers)]

    # Interpolate position embeddings
    height, width = vision_config.image_default_input_size
    num_patches = vision_config.image_num_pos - 1
    position_embedding = state_dict["embeddings"]["position_embedding"].pop("weight")
    position_embedding = interpolate_position_embeddings(
        position_embedding.unsqueeze(0),
        num_patches,
        position_embedding.shape[-1],
        vision_config.image_patch_size,
        height,
        width,
    )

    patch_embedding = state_dict["embeddings"]["patch_embedding"].pop(
        "weight"
    ).permute(0, 2, 3, 1).reshape(vision_config.image_emb_dim, -1)

    pre_ln = {
        "weight": state_dict["pre_layrnorm"].pop("weight"),
        "bias": state_dict["pre_layrnorm"].pop("bias"),
    }

    out = {
        "class_embedding": state_dict["embeddings"].pop("class_embedding"),
        "positional_embedding": position_embedding[0],
        "patch_embedding": dict(weight=patch_embedding),
        "pre_ln": pre_ln,
        "transformer": dict(resblocks=resblocks),
    }
    out = flatten_dict(out, sep=".")
    del state_dict["post_layernorm"]
    for k in flatten_dict(state_dict):
        raise ValueError("Unused parameter:", k)
    return out


def convert_state_dict_siglip(state_dict, vision_config: VisionBackboneConfig) -> Dict[str, Any]:
    state_dict = unflatten_dict(state_dict, sep=".")

    resblocks = {}
    for layer in range(vision_config.image_num_layers):
        layer_dict = state_dict["encoder"]["layers"][str(layer)]
        q, k, v, o = [
            layer_dict["self_attn"][f"{x}_proj"].pop("weight")
            for x in ["q", "k", "v", "out"]
        ]
        q_b, k_b, v_b, o_b = [
            layer_dict["self_attn"][f"{x}_proj"].pop("bias")
            for x in ["q", "k", "v", "out"]
        ]

        w1, w2 = [layer_dict["mlp"][f"{x}"].pop("weight") for x in ["fc1", "fc2"]]
        w1_b, w2_b = [layer_dict["mlp"][f"{x}"].pop("bias") for x in ["fc1", "fc2"]]

        mapped_layer_dict = {
            "attention": {
                "wq": dict(weight=q, bias=q_b),
                "wk": dict(weight=k, bias=k_b),
                "wv": dict(weight=v, bias=v_b),
                "wo": dict(weight=o, bias=o_b),
            },
            "feed_forward": {
                "w1": dict(weight=w1, bias=w1_b),
                "w2": dict(weight=w2, bias=w2_b), 
            },
            "attention_norm": {
                "weight": layer_dict["layer_norm1"].pop("weight"),
                "bias": layer_dict["layer_norm1"].pop("bias"),
            },
            "ffn_norm": {
                "weight": layer_dict["layer_norm2"].pop("weight"),
                "bias": layer_dict["layer_norm2"].pop("bias"),
            }
        }

        resblocks[str(layer)] = mapped_layer_dict

    # Interpolate position embeddings
    height, width = vision_config.image_default_input_size
    num_patches = vision_config.image_num_pos
    position_embedding = state_dict["embeddings"]["position_embedding"].pop("weight")
    position_embedding = interpolate_position_embeddings(
        position_embedding.unsqueeze(0),
        num_patches,
        position_embedding.shape[-1],
        vision_config.image_patch_size,
        height,
        width,
        num_prefix_tokens=0,
    )

    patch_embedding = state_dict["embeddings"]["patch_embedding"].pop(
        "weight"
    ).permute(0, 2, 3, 1).reshape(vision_config.image_emb_dim, -1)
    patch_embedding_b = state_dict["embeddings"]["patch_embedding"].pop("bias")

    out = {
        "positional_embedding": position_embedding[0],
        "patch_embedding": dict(weight=patch_embedding, bias=patch_embedding_b),
        "transformer": dict(resblocks=resblocks),
    }
    out = flatten_dict(out, sep=".")
    del state_dict["post_layernorm"]
    del state_dict["head"]
    for k in flatten_dict(state_dict):
        raise ValueError("Unused parameter:", k)
    return out


def convert_state_dict_dino(state_dict, vision_config: VisionBackboneConfig) -> Dict[str, Any]:
    state_dict = unflatten_dict(state_dict, sep=".")

    resblocks = {}
    for layer in range(vision_config.image_num_layers):
        layer_dict = state_dict["encoder"]["layer"][str(layer)]
        q, k, v = [
            layer_dict["attention"]["attention"][f"{x}"].pop("weight")
            for x in ["query", "key", "value"]
        ]
        q_b, k_b, v_b = [
            layer_dict["attention"]["attention"][f"{x}"].pop("bias")
            for x in ["query", "key", "value"]
        ]
        o = layer_dict["attention"]["output"]["dense"].pop("weight")
        o_b = layer_dict["attention"]["output"]["dense"].pop("bias")

        w1, w2 = [layer_dict["mlp"][f"{x}"].pop("weight") for x in ["fc1", "fc2"]]
        w1_b, w2_b = [layer_dict["mlp"][f"{x}"].pop("bias") for x in ["fc1", "fc2"]]

        mapped_layer_dict = {
            "attention": {
                "wq": dict(weight=q, bias=q_b),
                "wk": dict(weight=k, bias=k_b),
                "wv": dict(weight=v, bias=v_b),
                "wo": dict(weight=o, bias=o_b),
            },
            "feed_forward": {
                "w1": dict(weight=w1, bias=w1_b),
                "w2": dict(weight=w2, bias=w2_b), 
            },
            "attention_norm": {
                "weight": layer_dict["norm1"].pop("weight"),
                "bias": layer_dict["norm1"].pop("bias"),
            },
            "ffn_norm": {
                "weight": layer_dict["norm2"].pop("weight"),
                "bias": layer_dict["norm2"].pop("bias"),
            },
            "lambda1": layer_dict["layer_scale1"].pop("lambda1"),
            "lambda2": layer_dict["layer_scale2"].pop("lambda1"),
        }

        resblocks[str(layer)] = mapped_layer_dict

    # Interpolate position embeddings
    height, width = vision_config.image_default_input_size
    num_patches = vision_config.image_num_pos - 1
    position_embedding = state_dict["embeddings"].pop("position_embeddings")
    position_embedding = interpolate_position_embeddings(
        position_embedding,
        num_patches,
        position_embedding.shape[-1],
        vision_config.image_patch_size,
        height,
        width,
    )

    patch_embedding = state_dict["embeddings"]["patch_embeddings"]["projection"].pop(
        "weight"
    ).permute(0, 2, 3, 1).reshape(vision_config.image_emb_dim, -1)
    patch_embedding_b = state_dict["embeddings"]["patch_embeddings"]["projection"].pop("bias")

    out = {
        "class_embedding": state_dict["embeddings"].pop("cls_token").reshape(-1),
        "positional_embedding": position_embedding[0],
        "patch_embedding": dict(weight=patch_embedding, bias=patch_embedding_b),
        "transformer": dict(resblocks=resblocks),
    }
    out = flatten_dict(out, sep=".")
    del state_dict["layernorm"]
    del state_dict["embeddings"]["mask_token"]
    for k in flatten_dict(state_dict):
        raise ValueError("Unused parameter:", k)
    return out


def convert_state_dict_olmoe(state_dict, config: ModelConfig, block_type: BlockType) -> Dict[str, Any]:
    state_dict = unflatten_dict(state_dict, sep=".")
    assert len(state_dict) == 2
    lmhead = state_dict["lm_head"]
    state_dict = state_dict["model"]

    blocks = {}
    for layer in range(config.n_layers):
        layer_dict = state_dict["layers"][str(layer)]
        q, k, v, o = [layer_dict["self_attn"][f"{k}_proj"].pop("weight") for k in ["q", "k", "v", "o"]]

        assert block_type == BlockType.moe

        router = layer_dict["mlp"]["gate"].pop("weight")
        mlp_gates = [layer_dict["mlp"]["experts"][str(i)]["gate_proj"].pop("weight") for i in range(config.moe_num_experts)]
        mlp_ups = [layer_dict["mlp"]["experts"][str(i)]["up_proj"].pop("weight") for i in range(config.moe_num_experts)]
        mlp_downs = [layer_dict["mlp"]["experts"][str(i)]["down_proj"].pop("weight").t() for i in range(config.moe_num_experts)]

        ffn = {
            "router": {
                "layer": dict(weight=router),
            },
            "experts": {
                "mlp": dict(
                    w1=torch.cat(mlp_gates, 0),
                    v1=torch.cat(mlp_ups, 0),
                    w2=torch.cat(mlp_downs, 0),
                ),
            }
        }

        mapped_layer_dict = {
            "ffn": ffn,
            "ff_norm": {
                "weight": layer_dict[f"post_attention_layernorm"].pop("weight")
            },
            "attn_norm": {
                "weight": layer_dict[f"input_layernorm"].pop("weight")
            },
            "att_proj": dict(
                weight=torch.cat((q, k, v), dim=0),
            ),
            "attn_out": dict(weight=o),
            "q_norm": {
                "weight": layer_dict["self_attn"]["q_norm"].pop("weight"),
            },
            "k_norm": {
                "weight": layer_dict["self_attn"]["k_norm"].pop("weight"),
            },
        }

        blocks[str(layer)] = mapped_layer_dict

    out = flatten_dict(dict(transformer=dict(blocks=blocks)), sep=".")
    assert list(lmhead) == ["weight"]
    out.update({
        "transformer.wte.embedding": state_dict["embed_tokens"].pop("weight"),
        "transformer.ln_f.weight": state_dict["norm"].pop("weight"),
        "transformer.ff_out.weight": lmhead.pop("weight"),
    })
    for k in flatten_dict(state_dict):
        raise ValueError("Unused parameter:", k)
    return out


def convert_state_dict_olmo_1024_preview(state_dict, config: ModelConfig, block_type: BlockType) -> Dict[str, Any]:
    state_dict = unflatten_dict(state_dict, sep=".")
    assert len(state_dict) == 2
    lmhead = state_dict["lm_head"]
    state_dict = state_dict["model"]

    blocks = {}
    for layer in range(config.n_layers):
        layer_dict = state_dict["layers"][str(layer)]
        q, k, v, o = [layer_dict["self_attn"][f"{k}_proj"].pop("weight") for k in ["q", "k", "v", "o"]]
        mlp_gate = layer_dict["mlp"]["gate_proj"].pop("weight")
        mlp_up = layer_dict["mlp"]["up_proj"].pop("weight")
        mlp_down = layer_dict["mlp"]["down_proj"].pop("weight")

        assert block_type == BlockType.sequential

        mapped_layer_dict = {
            "ff_proj": {
                "weight": torch.cat([mlp_up, mlp_gate], 0)
            },
            "ff_out": {
                "weight": mlp_down
            },
            "ff_norm": {
                "weight": layer_dict[f"post_feedforward_layernorm"].pop("weight")
            },
            "attn_norm": {
                "weight": layer_dict[f"post_attention_layernorm"].pop("weight")
            },
            "att_proj": dict(
                weight=torch.cat((q, k, v), dim=0),
            ),
            "attn_out": dict(weight=o),
            "q_norm": {
                "weight": layer_dict["self_attn"]["q_norm"].pop("weight"),
            },
            "k_norm": {
                "weight": layer_dict["self_attn"]["k_norm"].pop("weight"),
            },
        }

        blocks[str(layer)] = mapped_layer_dict

    out = flatten_dict(dict(transformer=dict(blocks=blocks)), sep=".")
    assert list(lmhead) == ["weight"]
    out.update({
        "transformer.wte.embedding": state_dict["embed_tokens"].pop("weight"),
        "transformer.ln_f.weight": state_dict["norm"].pop("weight"),
        "transformer.ff_out.weight": lmhead.pop("weight"),
    })
    for k in flatten_dict(state_dict):
        raise ValueError("Unused parameter:", k)
    return out


def convert_state_dict_qwen2(state_dict, config: ModelConfig, block_type: BlockType) -> Dict[str, Any]:
    state_dict = unflatten_dict(state_dict, sep=".")
    assert len(state_dict) == 2
    lmhead = state_dict["lm_head"]
    state_dict = state_dict["model"]

    blocks = {}
    for layer in range(config.n_layers):
        layer_dict = state_dict["layers"][str(layer)]
        q, k, v, o = [layer_dict["self_attn"][f"{k}_proj"].pop("weight") for k in ["q", "k", "v", "o"]]
        if config.qkv_bias:
            q_b, k_b, v_b = [layer_dict["self_attn"][f"{k}_proj"].pop("bias") for k in ["q", "k", "v"]]
        else:
            q_b, k_b, v_b = None, None, None

        mlp_gate = layer_dict["mlp"]["gate_proj"].pop("weight")
        mlp_up = layer_dict["mlp"]["up_proj"].pop("weight")
        mlp_down = layer_dict["mlp"]["down_proj"].pop("weight")

        if block_type == BlockType.llama:
            mapped_layer_dict = {
                "q_proj": dict(weight=q, bias=q_b),
                "k_proj": dict(weight=k, bias=k_b),
                "v_proj": dict(weight=v, bias=v_b),
                "attn_out": dict(weight=o),
                "ff_norm": {
                    "weight": layer_dict[f"post_attention_layernorm"].pop("weight")
                },
                "attn_norm": {
                    "weight": layer_dict[f"input_layernorm"].pop("weight")
                },
                "ff_proj1": dict(weight=mlp_gate),
                "ff_proj2": dict(weight=mlp_up),
                "ff_out": dict(weight=mlp_down),
            }
        elif block_type == BlockType.sequential:
            mapped_layer_dict = {
                "ff_proj": {
                    "weight": torch.cat([mlp_up, mlp_gate], 0)
                },
                "ff_out": {
                    "weight": mlp_down
                },
                "ff_norm": {
                    "weight": layer_dict[f"post_attention_layernorm"].pop("weight")
                },
                "attn_norm": {
                    "weight": layer_dict[f"input_layernorm"].pop("weight")
                },
                "att_proj": dict(
                    weight=torch.cat((q, k, v), dim=0),
                    bias=None if q_b is None else torch.cat((q_b, k_b, v_b), dim=0)
                ),
                "attn_out": dict(weight=o),
            }
        else:
            raise NotImplementedError(block_type)
        blocks[str(layer)] = mapped_layer_dict

    out = flatten_dict(dict(transformer=dict(blocks=blocks)), sep=".")
    assert list(lmhead) == ["weight"]
    out.update({
        "transformer.wte.embedding": state_dict["embed_tokens"].pop("weight"),
        "transformer.ln_f.weight": state_dict["norm"].pop("weight"),
        "transformer.ff_out.weight": lmhead.pop("weight"),
    })
    for k in flatten_dict(state_dict):
        raise ValueError("Unused parameter:", k)
    return out


def get_default_load_path(model_name: str) -> str:
    default_load_path = DEFAULT_LOAD_PATHS[model_name]
    return "/".join(default_load_path.split("/")[1:])


CONVERT_FNS = {
    "openai": convert_state_dict_clip,
    "siglip": convert_state_dict_siglip,
    "dinov2_large_336": convert_state_dict_dino,
    "metaclip_l14_336": convert_state_dict_clip,
    "olmoe": convert_state_dict_olmoe,
    "olmo_1024_preview": convert_state_dict_olmo_1024_preview,
    "qwen2_7b": convert_state_dict_qwen2,
    "qwen2_72b": convert_state_dict_qwen2,
}


VIT_HF_SOURCES  = {
    "openai": "openai/clip-vit-large-patch14-336",
    "siglip": "google/siglip-so400m-patch14-384",
    "dinov2_large_336": "facebook/dinov2-large",
    "metaclip_l14_336": "facebook/metaclip-l14-fullcc2.5b",
}


LLM_HF_SOURCES = {
    "olmoe": "allenai/OLMoE-1B-7B-0924",
    "olmo_1024_preview": "allenai/OLMo-7B-1024-preview",
    "qwen2_7b": "Qwen/Qwen2-7B",
    "qwen2_72b": "Qwen/Qwen2-72B",
}


def main_vit(args: argparse.Namespace) -> None:
    hf_source = VIT_HF_SOURCES[args.model]
    cfg = ModelConfig(vision_backbone=VISION_BACKBONES[args.model])
    cfg.init_device = 'cpu'
    v_cfg = cfg.vision_backbone
    convert_fn = CONVERT_FNS[args.model]

    output_path = str(Path(args.data_dir).joinpath(get_default_load_path(args.model)))
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)

    logging.info(f"Convert model {args.model} to olmo format and save to {output_path}...")

    logging.info(f"Loading model from {hf_source}...")

    model = AutoModel.from_pretrained(
        hf_source,
        torch_dtype=torch.float32,
        cache_dir=args.cache_dir,
    )
    if isinstance(model, (CLIPModel, SiglipModel)):
        model = model.vision_model

    state_dict = model.state_dict()

    logging.info("Converting...")

    vit_state_dict = convert_fn(state_dict, v_cfg)
    
    logging.info("Saving...")
    torch.save(vit_state_dict, output_path)


def main_llm(args: argparse.Namespace) -> None:
    hf_source = LLM_HF_SOURCES[args.model]
    cfg = LLMS[args.model]
    cfg.init_device = 'cpu'
    convert_fn = CONVERT_FNS[args.model]

    output_path = str(Path(args.data_dir).joinpath(get_default_load_path(args.model)))
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)

    logging.info(f"Convert model {args.model} to olmo format and save to {output_path}...")

    logging.info(f"Loading model from {hf_source}...")

    model = AutoModelForCausalLM.from_pretrained(
        hf_source,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
        revision="fp32" if args.model == "olmoe" else "main",
    )

    state_dict = model.state_dict()

    logging.info("Converting...")

    olmo_state_dict = convert_fn(state_dict, cfg, cfg.block_type)

    logging.info("Saving...")
    torch.save(olmo_state_dict, output_path)


def main(args: argparse.Namespace) -> None:
    prepare_cli_environment()

    if args.data_dir is None:
        if "MOLMO_DATA_DIR" not in os.environ:
            raise ValueError("Either `data_dir` or env variable MOLMO_DATA_DIR must be set")
        args.data_dir = os.environ["MOLMO_DATA_DIR"]
        logging.info(f"Defaulting to data dir {args.data_dir}.")

    if args.model in VISION_BACKBONES:
        main_vit(args)
    elif args.model in LLMS:
        main_llm(args)
    else:
        raise ValueError(f"Unknown model {args.model}")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="convert hf vit/llm to molmo format script")
    parser.add_argument(
        "model",
        type=str,
        help="Model to be converted",
        choices=list(CONVERT_FNS.keys()),
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Needed to save converted model weights. It is a directory",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Directory to save HF parameters",
    )

    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args)
