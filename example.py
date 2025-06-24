from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import os
import torch
import json
import spacy
import argparse
from openai import OpenAI

from utils.prompts import *
from utils.scalecap_utils import scalecap_forward



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="caption generator using VLM and LLM api")
    parser.add_argument("--llm-port", type=int, default=21000, required=False, help="LLM service port")
    parser.add_argument("--vlm-port", type=int, default=31000, required=False, help="VLM service port")
    parser.add_argument("--second-filter", action="store_true", help="whether do the second contrastive sentence rating")

    args = parser.parse_args()

    threshold = 0
    image_path = "/path/image.png"
    second_filter = args.second_filter

    # Set OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    vlm_api_base = f"http://127.0.0.1:{args.vlm_port}/v1"

    # Load VLM agent
    vlm_client = OpenAI(
        api_key=openai_api_key,
        base_url=vlm_api_base,
    )

    # Set OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    llm_api_base = f"http://127.0.0.1:{args.llm_port}/v1"

    # Load LLM agent
    llm_client = OpenAI(
        api_key=openai_api_key,
        base_url=llm_api_base,
    )

    # Load VLM
    vlm_model_path = "/path/models--Qwen--Qwen2-VL-7B-Instruct/"
    vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
        vlm_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    vlm_processor = AutoProcessor.from_pretrained(vlm_model_path, max_pixels=4096*28*28)
    vlm_tokenizer = vlm_processor.tokenizer
    vlm_special_tokens = json.load(open(os.path.join(vlm_model_path, "tokenizer_config.json"), "r"))["additional_special_tokens"]

    # Load nlp model
    nlp_model = spacy.load("en_core_web_md")
    wanted_mapping = {'ADJ':1, 'ADP':0, 'ADV':1, 'AUX':0, 'CONJ':0, 'CCONJ':0, 'DET':0, 'INTJ':1, 'NOUN':1, 'NUM':1, 'PART':0, 'PRON':0, 'PROPN':1, 'PUNCT':0, 'SCONJ':0, 'SYM':0, 'VERB':1, 'X':0, 'SPACE':0}
    wanted_mapping_list = [s for s in wanted_mapping.keys() if wanted_mapping[s]]
    no_wanted_mapping = len(wanted_mapping.keys()) == len(wanted_mapping_list)
    wanted_mapping_str = "_".join(wanted_mapping_list) if not no_wanted_mapping else "all"


    golden_params = {                 # params for func get_golden_sentences
        "vlm_model": vlm_model,
        "nlp_model": nlp_model,
        "vlm_processor": vlm_processor,
        "vlm_tokenizer": vlm_tokenizer,
        "threshold": threshold,
        "wanted_mapping_str": wanted_mapping_str,
        "wanted_mapping": wanted_mapping
    }
    client_params = {                 # params for api
        "llm_client": llm_client,
        "vlm_client": vlm_client,
    }
    initial_caption, final_caption = scalecap_forward(image_path, **golden_params, **client_params, second_filter = second_filter)
    print("Initial caption:", initial_caption)
    print("Final caption after scalecap:", final_caption)