import os
import sys
import gc
import re
import time
import json
import logging
import argparse

from io import BytesIO
from tqdm import tqdm
from PIL import Image

import torch
import transformers

from vllm import LLM, SamplingParams
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    GenerationConfig,
    BitsAndBytesConfig,
)


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("select_alignment_training_prompts")


def parse_llm_output(text, aid_text='assistant'):
    text = text.split(aid_text)[-1]
    # explain_pattern = re.compile(r'\{"explain": .*\}')
    # explain_match = re.findall(explain_pattern, text)

    results_pattern = re.compile(
        r'\{"results": \{"Object": .*Spatial": .*\}\}')
    results_match = re.findall(results_pattern, text)
    
    # if len(results_match) != 0 and len(explain_match) != 0:
    #     return explain_match[-1], results_match[-1]
    # elif len(results_match) != 0:
    #     return '{"explain": ""}', results_match[-1]
    # else:
    #     return None, None
    if len(results_match) != 0:
        return results_match[-1]
    return None


def main(args):
    model_path = args.model_path
    dataset_path = args.dataset_path
    batch_size = args.batch_size
    chunks = args.chunks
    offset = args.offset
    save_path = args.save_path
    prompt_template_path = args.prompt_template_path
    aid_text = 'assistant'

    
    # load model
    logger.info(f"Loading model from {model_path}...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        # quantization_config=quantization_config,
        attn_implementation="flash_attention_2",
        device_map="auto",
    ).eval()  # .to('cuda')
    model.tie_weights()

    # load processor
    logger.info(f"Loading processor from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path)
    terminators = [
        processor.eos_token_id,
        processor.convert_tokens_to_ids("<|eot_id|>")
    ]
    processor.pad_token_id = processor.eos_token_id
    processor.pad_token = processor.eos_token
    processor.padding_side = "left"

    # load dataset
    logger.info(f"Loading dataset from {dataset_path}...")
    orig_dataset = load_dataset(dataset_path)
    t2i_prompts = []
    for data in tqdm(orig_dataset['train']):
        if not data['has_label']:
            continue
        t2i_prompts.append(data['caption'])
    start_index = len(t2i_prompts)//chunks * offset
    end_index = len(t2i_prompts)//chunks * (offset+1)
    t2i_prompts = t2i_prompts[:end_index]
    
    # load prompt template
    with open(prompt_template_path, 'r') as rf:
        prompt_template = rf.read()
        
    # generation config
    gen_config = GenerationConfig(
        temperature=1,
        top_p=1,
        do_sample=False,
        num_beams=1,
        max_new_tokens=384,
        eos_token_id=terminators,
    )
    dataset = []
    for idx in tqdm(range(start_index, end_index, batch_size)):
        # start_time = time.time()
        batch_data = t2i_prompts[idx:idx+batch_size]
        prompts = [prompt_template.format(caption) for caption in batch_data]
        inputs = processor(prompts, return_tensors="pt", padding=True).to('cuda')
        output = model.generate(**inputs, generation_config=gen_config)
        output = processor.batch_decode(output, skip_special_tokens=True)
        # print(time.time() - start_time)
        
        if len(output) != len(batch_data): continue
        for o_idx, o in enumerate(output):
            try:
                results = parse_llm_output(o, aid_text)
                # if results == None: continue
                data = {
                    "idx": idx+o_idx,
                    "caption": batch_data[o_idx],
                    "explain": o.split(aid_text)[-1],
                    "results": json.loads(results)['results']
                }
                dataset.append(data)
            except:
                logger.info(f"Can not parsing the output of index {idx+o_idx}..")
        if (idx+batch_size-start_index) % 2048 == 0 and idx > 0:
            logger.info(
                f"Saving datapoints from index {idx+batch_size-2048} to {idx+batch_size}...")
            with open(save_path, 'a') as af:
                for data in dataset:
                    line = json.dumps(data, ensure_ascii=False)
                    af.write(line + "\n")
            dataset = []
    
    logger.info(f"Saving datapoints from index {end_index-2048} to {end_index}...")
    with open(save_path, 'a') as af:
        for data in dataset:
            line = json.dumps(data, ensure_ascii=False)
            af.write(line + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "-m", type=str,
                        default="/nas/shared/NLP_A100/ycdu/models/meta-llama/Meta-Llama-3-70B-Instruct")
    parser.add_argument("--dataset-path", "-dp", type=str)
    parser.add_argument("--save-path", "-sp", type=str)
    parser.add_argument("--prompt-template-path", type=str)
    parser.add_argument("--batch-size", "-bsz", type=int, default=64)
    parser.add_argument("--chunks", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    main(args)
