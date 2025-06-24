from qwen_vl_utils import process_vision_info
import torch
from transformers.cache_utils import Cache, DynamicCache
import torch.nn.functional as F
import base64
from utils.prompts import *
import concurrent.futures

def map_token_to_pos(tokens, output_text, nlp_model):
    doc = nlp_model(output_text)
    words_pos = [(word.text, word.pos_) for word in doc]
    tokens = ["".join(token.split(" ")) for token in tokens]
    words_pos = [("".join(word_pos[0].split(" ")), word_pos[1]) for word_pos in words_pos]

    mapped = []
    token_index = -1
    offset = 0

    for i in range(len(tokens)):
        if tokens[i] == "<0x0A>":
            tokens[i] = "\n"

    for i in range(len(words_pos)):
        current_word = words_pos[i][0]
        current_pos = words_pos[i][1]
        buffer = ""
        if offset > 0:
            offset -= 1
            continue

        while buffer != current_word:
            if len(buffer) <= len(current_word):
                token_index += 1
                buffer += tokens[token_index]
                mapped.append(current_pos)
            else:
                offset += 1
                current_word += words_pos[i+offset][0]

    return mapped


def prepare_input(question, vlm_processor ,image_path=None):
    if type(question) is list:
        if image_path is not None:
            text_list = []
            image_inputs_list = []
            for p in question:
                messages = [
                    {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": p}]}
                ]
                text = vlm_processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                text_list.append(text)
    
                image_inputs, _ = process_vision_info(messages)
                image_inputs_list.append(image_inputs)
    
            inputs = vlm_processor(
                text=text_list,
                images=image_inputs_list,
                videos=None,
                padding=True,
                padding_side='left', 
                truncation=True,
                return_tensors="pt",
            )
        else:
            text_list = []
            for p in question:
                messages = [
                    {"role": "user", "content": [{"type": "text", "text": p}]}
                ]
                text = vlm_processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                text_list.append(text)
    
            inputs = vlm_processor(
                text=text_list,
                padding=True,
                padding_side='left', 
                truncation=True,
                return_tensors="pt",
            )
    else:
        if image_path is not None:
            messages = [
                {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": question}]}
            ]
            text = vlm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = vlm_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        else:
            messages = [
                {"role": "user", "content": [{"type": "text", "text": question}]}
            ]
            text = vlm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = vlm_processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            )
    return inputs


def get_golden_sentences(prompt, image_path, output_text, vlm_model, nlp_model, vlm_processor, vlm_tokenizer, threshold, wanted_mapping_str, wanted_mapping):
    
    if not isinstance(output_text, list):
        output_text = [output_text]

    output_ids = [
        torch.LongTensor(vlm_tokenizer.encode(t)).to(vlm_model.device)
        for t in output_text
        ]

    # 1. ori_output_logits 
    ori_inputs = prepare_input(prompt, vlm_processor, image_path)
    ori_inputs = ori_inputs.to(vlm_model.device)

    # cat input and output
    concat_list = [
        torch.cat([ori_inputs.data["input_ids"][i], output_ids[i]], dim=0).to(vlm_model.device)
        for i in range(len(output_ids))
    ]
    # pad and form batch
    max_len = max(t.size(0) for t in concat_list)   # max length in the batch
    cur_ori_input_ids = torch.stack([
        F.pad(t, (max_len - t.size(0), 0), value=vlm_processor.tokenizer.pad_token_id)   # (left_pad)
        for t in concat_list
    ]).to(vlm_model.device)
    

    ori_inputs.data["input_ids"] = cur_ori_input_ids
    ori_inputs.data["attention_mask"] = torch.ones_like(cur_ori_input_ids).to(cur_ori_input_ids.device)
    ori_inputs.data['cache_position'] = torch.arange(cur_ori_input_ids.shape[-1]).to(cur_ori_input_ids.device)
    ori_inputs.data['past_key_values'] = DynamicCache()
    ori_inputs.data['use_cache'] = True

    with torch.inference_mode():
        cur_ori_outputs = vlm_model.forward(**ori_inputs)

    ori_logits_list = cur_ori_outputs.logits

    # 2. biased_output_probs
    biased_inputs = prepare_input(prompt, vlm_processor, None)
    biased_inputs = biased_inputs.to(vlm_model.device)

    # cat input and output
    concat_list = [
        torch.cat([biased_inputs.data["input_ids"][i], output_ids[i]], dim=0).to(vlm_model.device)
        for i in range(len(output_ids))
    ]
    # pad and form batch
    max_len = max(t.size(0) for t in concat_list)   # 所有样本的最大总长度
    cur_biased_input_ids = torch.stack([
        F.pad(t, (max_len - t.size(0), 0), value=vlm_processor.tokenizer.pad_token_id)   # (left_pad, right_pad)
        for t in concat_list
    ]).to(vlm_model.device)

    biased_inputs.data["input_ids"] = cur_biased_input_ids
    biased_inputs.data["attention_mask"] = torch.ones_like(cur_biased_input_ids).to(cur_biased_input_ids.device)
    biased_inputs.data['cache_position'] = torch.arange(cur_biased_input_ids.shape[-1]).to(cur_biased_input_ids.device)
    biased_inputs.data['past_key_values'] = DynamicCache()
    biased_inputs.data['use_cache'] = True

    with torch.inference_mode():
        cur_biasd_outputs = vlm_model.forward(**biased_inputs)

    biased_logits_list = cur_biasd_outputs.logits
    
    # 3. process each response to filter hallucinated sentence
    all_golden_sentences = []
    for j in range(len(biased_inputs.data["input_ids"])):
        ori_output_probs, biased_output_probs = [], []

        cur_ori_logits_list = ori_logits_list[j][-len(output_ids[j])-1:-1]
        for logit, id in zip(cur_ori_logits_list.tolist(), output_ids[j].tolist()):
            ori_output_probs.append(logit[id])

        cur_biased_logits_list = biased_logits_list[j][-len(output_ids[j])-1:-1]
        for logit, id in zip(cur_biased_logits_list.tolist(), output_ids[j].tolist()):
            biased_output_probs.append(logit[id])
        

        tokens = [vlm_processor.decode(id) for id in output_ids[j].tolist()]
        prob_gaps = [s1 - s2 for s1, s2 in zip(ori_output_probs, biased_output_probs)]
        
        if wanted_mapping_str != "all":
            try:
                tokens_type = map_token_to_pos(tokens, output_text, nlp_model)
                assert len(tokens) == len(tokens_type) == len(prob_gaps)
                for i in range(len(prob_gaps)):
                    prob_gaps[i] *= wanted_mapping[tokens_type[i]]
            except:
                pass
        
        
        punctuations = [",", ".", "?", "!", "\n", "\n\n", ".\n\n"]
        res_ids = output_ids[j].clone()
        res_ids = list(res_ids.cpu().numpy())
        res_ids_tokens = []
        for i in range(len(res_ids)):
            res_ids_tokens.append(vlm_processor.decode(res_ids[i]))
            res_ids[i] = -999 if res_ids_tokens[i] in punctuations[1:] and not (
                i == 0 or (res_ids_tokens[i-1] in punctuations[1:] and res_ids[i-1] != -999)) else res_ids[i]
            res_ids[i] = -888 if res_ids_tokens[i] in punctuations[:1] else res_ids[i]
        
        numbers = [[str(i)] for i in range(10)]
        numbers_ids = [vlm_tokenizer.convert_tokens_to_ids(number) for number in numbers]
        for idx in range(len(res_ids) - 1):
            if res_ids[idx] == -999 and (res_ids[idx+1] in numbers_ids or (idx > 0 and res_ids[idx-1] in numbers_ids)):
                res_ids[idx] = vlm_tokenizer.convert_tokens_to_ids(".")
        
        tmp_list, tmp_score, positive_ids_list, negative_ids_list, positive_sent_list, negtive_sent_list = [], [], [], [], [], []
        for id, score in zip(res_ids, prob_gaps):
            if id == -999:
                if len(tmp_list) > 0 and max(tmp_score) > threshold:
                    positive_ids_list.append(tmp_list)
                elif len(tmp_list) > 0 and max(tmp_score) <= threshold:
                    negative_ids_list.append(tmp_list)
                tmp_list = []
                tmp_score = []
                if len(positive_ids_list) > 0:
                    positive_sent_list.append(positive_ids_list)
                elif len(negative_ids_list) > 0:
                    negtive_sent_list.append(negative_ids_list)
                positive_ids_list = []
                negative_ids_list = []
            elif id == -888:
                if len(tmp_list) > 0 and max(tmp_score) > threshold:
                    positive_ids_list.append(tmp_list)
                elif len(tmp_list) > 0 and max(tmp_score) <= threshold:
                    negative_ids_list.append(tmp_list)
                tmp_list = []
                tmp_score = []
            else:
                tmp_list.append(id)
                tmp_score.append(score)
        
        sentences = [",".join(vlm_processor.batch_decode(positive_sent, skip_special_tokens=True)) for positive_sent in positive_sent_list]
        golden_sentences_list = []
        for sent in sentences:
            if sent.startswith(" "):
                golden_sentences_list.append(sent[1:])
            else:
                golden_sentences_list.append(sent)
        golden_sentences = []
        for golden_sent in golden_sentences_list:
            for sent in golden_sent.split("\n"):
                if len(sent) > 0:
                    golden_sentences.append(sent)
        
        all_golden_sentences.append(golden_sentences)
        
    return all_golden_sentences



def vlm_client_qa(prompt, image_path, vlm_client, max_new_tokens=512):
    assert not type(prompt) is list
    with open(image_path, "rb") as f:
        encoded_image = base64.b64encode(f.read())
    encoded_image_text = encoded_image.decode("utf-8")
    base64_qwen = f"data:image;base64,{encoded_image_text}"

    chat_response = vlm_client.chat.completions.create(
        model="qwen2-vlm",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": base64_qwen
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        temperature=0.0,
        max_tokens=max_new_tokens,
        extra_body={
            "repetition_penalty": 1.0,
        },
    )
    # print("Chat response:", chat_response)
    output_text = chat_response.choices[0].message.content
    return output_text


def multi_threaded_vlm_client_qa(prompt, image_path, vlm_client, max_new_tokens=512):
    assert type(prompt) is list
    results = [None] * len(prompt)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(vlm_client_qa, p, image_path, vlm_client, max_new_tokens): idx
                    for idx, p in enumerate(prompt)}
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            result = future.result()
            results[idx] = result
    return results


def llm_client_qa(prompt, llm_client, max_new_tokens=512):
    assert not type(prompt) is list
    chat_response = llm_client.chat.completions.create(
        model="qwen2-llm",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=max_new_tokens,
        extra_body={
            "repetition_penalty": 1.05,
        },
    )
    # print("Chat response:", chat_response)
    output_text = chat_response.choices[0].message.content
    return output_text


def multi_threaded_llm_client_qa(prompt, llm_client, max_new_tokens=512):
    assert type(prompt) is list
    results = [None] * len(prompt)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(llm_client_qa, p, llm_client, max_new_tokens): idx
                    for idx, p in enumerate(prompt)}
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            result = future.result()
            results[idx] = result
    return results


def scalecap_forward(image_path, vlm_model, nlp_model, vlm_processor, vlm_tokenizer, threshold, wanted_mapping_str, wanted_mapping, llm_client, vlm_client, second_filter = False):
    # Step 1: Get original caption
    qs = VLM_PROMPT_1
    initial_caption = vlm_client_qa(qs, image_path, vlm_client, max_new_tokens=1024)

    # Step 2: Get golden sentences | first contrastive sentence rating 
    qs = VLM_PROMPT_1
    golden_params = {
        "vlm_model": vlm_model,
        "nlp_model": nlp_model,
        "vlm_processor": vlm_processor,
        "vlm_tokenizer": vlm_tokenizer,
        "threshold": threshold,
        "wanted_mapping_str": wanted_mapping_str,
        "wanted_mapping": wanted_mapping
    }
    golden_sentences = get_golden_sentences(qs, image_path, initial_caption, **golden_params)[0]
    golden_sentences = golden_sentences if len(golden_sentences) > 0 else [initial_caption]

    # Step 3: Create instructions from LLM
    qs1 = [LLM_PROMPT_1.format(golden_sentence) for golden_sentence in golden_sentences]
    response = multi_threaded_llm_client_qa(qs1, llm_client, max_new_tokens=512)

    instructions_list = []
    for ins in response:
        ins = ins[:ins.rfind('.')+1]
        ins = list(set(ins.split("\n")))
        ins = [t.split(".")[0]+"." for t in ins]
        ins = [t for t in ins if t.startswith("Describe more details about")]
        instructions_list.append(ins)

    # Step 4: Obtain object/position details from VLM
    detailed_obj_response_list = []
    detailed_pos_response_list = []
    all_obj_instructions = []
    obj_instructions, pos_instructions, cur_golden_sentences, cur_nums = [], [], [], []
    gathered_golden_sents = "\n".join(golden_sentences)

    assert len(golden_sentences) == len(instructions_list)
    for sent_idx in range(len(golden_sentences)):
        golden_sentence = golden_sentences[sent_idx]
        instructions = instructions_list[sent_idx]
        instructions = list(set(instructions))

        object_instructions = [ins for ins in instructions if not ins in all_obj_instructions] # keep only one instr for the same object
        all_obj_instructions += object_instructions 
        obj_instructions += object_instructions
        cur_golden_sentences.append(golden_sentence + ".")
        cur_nums.append(len(object_instructions))  # number of instrs for each golden_sentence
        
        # accumulate for multithread
        if len(obj_instructions) >= 10 or (len(obj_instructions) > 0 and sent_idx == len(golden_sentences) - 1):
            pos_instructions = [
                "Describe more details about the position of" + ins.split(
                    "Describe more details about")[-1] for ins in obj_instructions
            ]

            assert len(obj_instructions) == len(pos_instructions)
            try:
                cur_obj_response = multi_threaded_vlm_client_qa(obj_instructions, image_path, vlm_client, max_new_tokens=512)
                cur_pos_response = multi_threaded_vlm_client_qa(pos_instructions, image_path, vlm_client, max_new_tokens=512)
            
            # Reduce the number of concurrent threads if an error occurs. 
            except:
                i = 1
                obj_ins = obj_instructions[5*(i-1):5*i]
                pos_ins = pos_instructions[5*(i-1):5*i]
                cur_obj_response, cur_pos_response = [], []
                while len(obj_ins) > 0 and len(pos_ins) > 0:
                    obj_response = multi_threaded_vlm_client_qa(obj_ins, image_path, vlm_client, max_new_tokens=512)
                    pos_response = multi_threaded_vlm_client_qa(pos_ins, image_path, vlm_client, max_new_tokens=512)
                    cur_obj_response += obj_response
                    cur_pos_response += pos_response

                    i += 1
                    obj_ins = obj_instructions[5*(i-1):5*i]
                    pos_ins = pos_instructions[5*(i-1):5*i]
            
            # whether do the second contrastive sentence rating
            if second_filter:
                cur_obj_response_golden_list = []
                cur_obj_response_golden_all = get_golden_sentences(obj_instructions, image_path, cur_obj_response, **golden_params)
                for o_i,cur_obj_response_golden in enumerate(cur_obj_response_golden_all):
                    cur_obj_response_golden = " ".join([s + "." for s in cur_obj_response_golden]) if len(cur_obj_response_golden) > 0 else cur_obj_response[o_i]
                    cur_obj_response_golden_list.append(cur_obj_response_golden)

                cur_pos_response_golden_list = []
                cur_pos_response_golden_all = get_golden_sentences(pos_instructions, image_path, cur_pos_response, **golden_params)
                for p_i,cur_pos_response_golden in enumerate(cur_pos_response_golden_all):
                    cur_pos_response_golden = " ".join([s + "." for s in cur_pos_response_golden]) if len(cur_pos_response_golden) > 0 else cur_pos_response[p_i]
                    cur_pos_response_golden_list.append(cur_pos_response_golden)
            else:
                cur_obj_response = [" ".join(res.split("\n")) for res in cur_obj_response]
                cur_pos_response = [" ".join(res.split("\n")) for res in cur_pos_response]

                cur_obj_response_golden_list = cur_obj_response
                cur_pos_response_golden_list = cur_pos_response
                

            # organize each golden sentence with response
            offset = 0
            for s_i in range(len(cur_golden_sentences)):
                detailed_obj_response_list.append("\nSentence: " + cur_golden_sentences[s_i])
                detailed_pos_response_list.append("\nSentence: " + cur_golden_sentences[s_i])
                for _ in range(cur_nums[s_i]):
                    obj_name = obj_instructions[offset].split("Describe more details about ")[-1][:-1]
                    detailed_obj_response_list.append("Details about {}: ".format(obj_name) + cur_obj_response_golden_list[offset])
                    detailed_pos_response_list.append("Position about {}: ".format(obj_name) + cur_pos_response_golden_list[offset])
                    offset += 1
            assert offset == len(cur_obj_response_golden_list)
            obj_instructions, pos_instructions, cur_golden_sentences, cur_nums = [], [], [], []

    detailed_obj_response = "\n".join(detailed_obj_response_list)
    detailed_pos_response = "\n".join(detailed_pos_response_list)
    
    # Here is an example of detailed_obj_response for an intuitive understanding
    '''
    Sentence: The man is dressed in a light blue long-sleeve shirt and blue jeans, and he is wearing black shoes.
    Details about the light blue long-sleeve shirt: The light blue long-sleeve shirt is worn by the man sitting on the bench. It appears to be a casual, comfortable fit, suitable for a relaxed day outdoors.
    Details about the black shoes: The man is wearing black leather shoes. They appear to be well-fitted and polished, suggesting that he takes care of his footwear.
    Details about the blue jeans: The man is wearing blue jeans that appear to be of a standard fit, with a slightly faded look. The jeans are dark blue in color and cover his legs from the waist down to his ankles.

    Sentence: He has a mustache and is sitting with his hands clasped together in his lap, possibly lost in thought or observing something in the distance.
    ......
    '''

    # Step 5: Organize the detailed object description via LLM
    qs1 = LLM_PROMPT_2.format(detailed_obj_response)
    detailed_obj_caption = llm_client_qa(qs1, llm_client, max_new_tokens=4096)

    # Step 6: Organize the detailed position description via LLM
    qs1 = LLM_PROMPT_3.format(detailed_pos_response)
    detailed_pos_caption = llm_client_qa(qs1, llm_client, max_new_tokens=4096)

    # Step 7: Organize the final description via LLM
    qs1 = LLM_PROMPT_4.format(gathered_golden_sents, detailed_obj_caption, detailed_pos_caption)
    final_caption = llm_client_qa(qs1, llm_client, max_new_tokens=4096)
    
    return initial_caption, final_caption