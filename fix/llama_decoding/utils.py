import os 
from transformers import GenerationMixin
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from fix.llama_decoding.decoding import generate_with_wrong_code, greedy_search_with_wrong_code, prepare_inputs_for_generation
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import torch 


def augment_generate():
    # TODO: when use generate, should pass wrong code to it if using augmented model
    GenerationMixin.generate = generate_with_wrong_code
    GenerationMixin.greedy_search = greedy_search_with_wrong_code
    LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation