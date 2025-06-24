from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import os
import torch

import json
import spacy

import shortuuid
from tqdm import tqdm
from openai import OpenAI


from utils.prompts import *
from utils.scalecap_utils import scalecap_forward
from utils.scalecap_utils import llm_client_qa



def process_question(line, ans_file, cap_file, second_filter):
    question_idx = line["imgname"].split(".")[0]
    image_file = line["imgname"]
    question = line["query"]
    gt_answer = line["label"]

    if image_file in id2captions.keys():
        final_caption = id2captions[image_file]["final_caption"]
    else:
        image_path = os.path.join(image_folder, image_file)

        golden_params = {                 # params for func get_golden_sentences
            "vlm_model": vlm_model,
            "nlp_model": nlp_model,
            "vlm_processor": vlm_processor,
            "vlm_tokenizer": vlm_tokenizer,
            "threshold": threshold,
            "wanted_mapping_str": wanted_mapping_str,
            "wanted_mapping": wanted_mapping
        }
        client_params = {
            "llm_client": llm_client,
            "vlm_client": vlm_client,
        }
        initial_caption, final_caption = scalecap_forward(image_path, **golden_params, **client_params, second_filter = second_filter)

    # Get the answer from LLM based on caption
    qs1 = LLM_PROMPT_5.format(final_caption, question)
    answer = llm_client_qa(qs1, llm_client, max_new_tokens=1024)
    print("---------------------------------")
    print(f"LLM Answer #{question_idx}:\n", answer)

    qs2 = LLM_PROMPT_7.format(question, answer)
    answer = llm_client_qa(qs2, llm_client, max_new_tokens=1024)
    print("---------------------------------")
    print(f"Brief LLM Answer #{question_idx}:\n", answer)

    ans_file.write(json.dumps({"question_id": question_idx,
                                "question": question,
                                "gt_answer": gt_answer, 
                                "prediction": answer}) + "\n")
    ans_file.flush()

    if not image_file in id2captions.keys():
        id2captions[image_file] = {"ori_caption": initial_caption, "final_caption": final_caption}

        cap_file.write(json.dumps({"image_file": image_file,
                                    "ori_caption": initial_caption,
                                    "final_caption": final_caption}) + "\n")
        cap_file.flush()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="caption generator using VLM and LLM api")
    parser.add_argument("--llm-port", type=int, default=21000, required=False, help="LLM service port")
    parser.add_argument("--vlm-port", type=int, default=31000, required=False, help="VLM service port")
    parser.add_argument("--second-filter", action="store_true", help="whether do the second contrastive sentence rating")
    args = parser.parse_args()
        
    threshold = 0
    model_name = "ScaleCap-VLM-7B-LLM-72B"

    vlm_model_path = "/path/models--Qwen--Qwen2-VL-7B-Instruct/"
    data_path = "/path/chartqa/test/test_human.json"
    image_folder = "/path/data/chartqa/test/png/"
    answers_file = f"./eval/chartqa/answers/{model_name}.jsonl"
    caption_file = f"./eval/chartqa/captions/{model_name}.jsonl"
    second_filter = args.second_filter

    questions = json.load(open(data_path, "r"))

    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    os.makedirs(os.path.dirname(caption_file), exist_ok=True)

    # Set OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    llm_api_base = f"http://127.0.0.1:{args.llm_port}/v1"

    # Load LLM agent
    llm_client = OpenAI(
        api_key=openai_api_key,
        base_url=llm_api_base,
    )

    # Set OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    vlm_api_base = f"http://127.0.0.1:{args.vlm_port}/v1"

    # Load VLM agent
    vlm_client = OpenAI(
        api_key=openai_api_key,
        base_url=vlm_api_base,
    )
    
    # Load VLM
    vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
        vlm_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    vlm_processor = AutoProcessor.from_pretrained(vlm_model_path, max_pixels=4096*28*28)
    vlm_tokenizer = vlm_processor.tokenizer
    vlm_special_tokens = json.load(open(vlm_model_path + "tokenizer_config.json", "r"))["additional_special_tokens"]
    
    # Load nlp model
    nlp_model = spacy.load("en_core_web_md")
    wanted_mapping = {'ADJ':1, 'ADP':0, 'ADV':1, 'AUX':0, 'CONJ':0, 'CCONJ':0, 'DET':0, 'INTJ':1, 'NOUN':1, 'NUM':1, 'PART':0, 'PRON':0, 'PROPN':1, 'PUNCT':0, 'SCONJ':0, 'SYM':0, 'VERB':1, 'X':0, 'SPACE':0}
    wanted_mapping_list = [s for s in wanted_mapping.keys() if wanted_mapping[s]]
    no_wanted_mapping = len(wanted_mapping.keys()) == len(wanted_mapping_list)
    wanted_mapping_str = "_".join(wanted_mapping_list) if not no_wanted_mapping else "all"
    
    # load caption already generated
    id2captions = {}
    if os.path.exists(caption_file):
        captions = [json.loads(line) for line in open(caption_file, "r")]
        for cap in captions:
            id2captions[cap["image_file"]] = {"ori_caption": cap["ori_caption"], "final_caption": cap["final_caption"]}

    ans_file = open(answers_file, "a")
    cap_file = open(caption_file, "a")

    # Main loop
    for line in tqdm(questions):
        process_question(line, ans_file, cap_file, second_filter)

    ans_file.close()
    cap_file.close()