import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union, Callable

import torch
import torch.distributed as dist
from transformers.generation.utils import GreedySearchDecoderOnlyOutput, GreedySearchEncoderDecoderOutput, GreedySearchOutput, GenerationConfig, GenerationMode, GenerateOutput
from transformers.utils import logging
from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
import copy
import inspect
if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer

logger = logging.get_logger(__name__)


@torch.no_grad()
def generate_with_wrong_code(
    self,
    wrong_code: Optional[torch.Tensor] = None,
    inputs: Optional[torch.Tensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    synced_gpus: Optional[bool] = None,
    assistant_model: Optional["PreTrainedModel"] = None,
    streamer: Optional["BaseStreamer"] = None,
    negative_prompt_ids: Optional[torch.Tensor] = None,
    negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:
    if synced_gpus is None:
        if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
            synced_gpus = True
        else:
            synced_gpus = False

    # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
    self._validate_model_class()

    # priority: `generation_config` argument > `model.generation_config` (the default generation config)
    if generation_config is None:
        # legacy: users may modify the model configuration to control generation. To trigger this legacy behavior,
        # two conditions must be met
        # 1) the generation config must have been created from the model config (`_from_model_config` field);
        # 2) the generation config must have seen no modification since its creation (the hash is the same).
        if self.generation_config._from_model_config and self.generation_config._original_object_hash == hash(
            self.generation_config
        ):
            new_generation_config = GenerationConfig.from_model_config(self.config)
            if new_generation_config != self.generation_config:
                warnings.warn(
                    "You have modified the pretrained model configuration to control generation. This is a"
                    " deprecated strategy to control generation and will be removed soon, in a future version."
                    " Please use and modify the model generation configuration (see"
                    " https://huggingface.co/docs/transformers/generation_strategies#default-text-generation-configuration )"
                )
                self.generation_config = new_generation_config
        generation_config = self.generation_config

    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
    generation_config.validate()
    self._validate_model_kwargs(model_kwargs.copy())

    # 2. Set generation parameters if not already defined
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
        if model_kwargs.get("attention_mask", None) is None:
            logger.warning(
                "The attention mask and the pad token id were not set. As a consequence, you may observe "
                "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
            )
        eos_token_id = generation_config.eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]
        logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
        generation_config.pad_token_id = eos_token_id

    # 3. Define model inputs
    # inputs_tensor has to be defined
    # model_input_name is defined if model-specific keyword input is passed
    # otherwise model_input_name is None
    # all model-specific keyword inputs are removed from `model_kwargs`
    inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    batch_size = inputs_tensor.shape[0]

    # 4. Define other model kwargs
    model_kwargs["output_attentions"] = generation_config.output_attentions
    model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
    # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
    # generating the first new token or not, and we only want to use the embeddings for the first new token)
    if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
        model_kwargs["use_cache"] = True
    else:
        model_kwargs["use_cache"] = generation_config.use_cache

    accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs

    if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
        model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
            inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
        )

    # decoder-only models should use left-padding for generation
    if not self.config.is_encoder_decoder:
        # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
        # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
        if (
            generation_config.pad_token_id is not None
            and len(inputs_tensor.shape) == 2
            and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
        ):
            logger.warning(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `padding_side='left'` when initializing the tokenizer."
            )

    if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        # if model is encoder decoder encoder_outputs are created
        # and added to `model_kwargs`
        model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name
        )

    # 5. Prepare `input_ids` which will be used for auto-regressive generation
    if self.config.is_encoder_decoder:
        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config.decoder_start_token_id,
            bos_token_id=generation_config.bos_token_id,
            device=inputs_tensor.device,
        )
    else:
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

    if streamer is not None:
        streamer.put(input_ids.cpu())

    # 6. Prepare `max_length` depending on other stopping criteria.
    input_ids_length = input_ids.shape[-1]
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    if generation_config.max_new_tokens is not None:
        if not has_default_max_length and generation_config.max_length is not None:
            logger.warning(
                f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                "Please refer to the documentation for more information. "
                "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
            )
        generation_config.max_length = generation_config.max_new_tokens + input_ids_length
    
    input_ids_to_check = input_ids_length + wrong_code.shape[-1] if wrong_code is not None else input_ids_length
    self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

    # 7. determine generation mode
    generation_mode = self._get_generation_mode(generation_config, assistant_model)

    if streamer is not None and (generation_config.num_beams > 1):
        raise ValueError(
            "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
        )

    if self.device.type != input_ids.device.type:
        warnings.warn(
            "You are calling .generate() with the `input_ids` being on a device type different"
            f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
            f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
            " Please make sure that you have put `input_ids` to the"
            f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
            " running `.generate()`.",
            UserWarning,
        )

    # 8. prepare distribution pre_processing samplers
    logits_processor = self._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
        model_kwargs=model_kwargs,
        negative_prompt_ids=negative_prompt_ids,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
    )

    # 9. prepare stopping criteria
    stopping_criteria = self._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria
    )
    # 10. go into different generation modes
    if generation_mode == GenerationMode.ASSISTED_GENERATION:
        if generation_config.num_return_sequences > 1:
            raise ValueError(
                "num_return_sequences has to be 1 when doing assisted generate, "
                f"but is {generation_config.num_return_sequences}."
            )
        if batch_size > 1:
            raise ValueError("assisted generate is only supported for batch_size = 1")
        if not model_kwargs["use_cache"]:
            raise ValueError("assisted generate requires `use_cache=True`")

        # 11. If the assistant model is an encoder-decoder, prepare its encoder outputs
        if assistant_model.config.is_encoder_decoder:
            assistant_model_kwargs = copy.deepcopy(model_kwargs)
            inputs_tensor, model_input_name, assistant_model_kwargs = assistant_model._prepare_model_inputs(
                inputs_tensor, assistant_model.generation_config.bos_token_id, assistant_model_kwargs
            )
            assistant_model_kwargs = assistant_model._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, assistant_model_kwargs, model_input_name
            )
            model_kwargs["assistant_encoder_outputs"] = assistant_model_kwargs["encoder_outputs"]

        # 12. run assisted generate
        return self.assisted_decoding(
            input_ids,
            assistant_model=assistant_model,
            do_sample=generation_config.do_sample,
            logits_processor=logits_processor,
            logits_warper=self._get_logits_warper(generation_config) if generation_config.do_sample else None,
            stopping_criteria=stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            output_scores=generation_config.output_scores,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )
    if generation_mode == GenerationMode.GREEDY_SEARCH:
        # 11. run greedy search
        return self.greedy_search(
            wrong_code,
            input_ids,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            output_scores=generation_config.output_scores,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )



def greedy_search_with_wrong_code(
    self,
    wrong_code: torch.LongTensor,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[GreedySearchOutput, torch.LongTensor]:
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    
    # TODO: now just hard coding the max_length, it should be set by the upper trace function[generation_config.max_length]
    max_length = 5000

    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    this_peer_finished = False  # used by synced_gpus only

    split_indices = (wrong_code == 13).nonzero(as_tuple=True)[1]
    wrong_code_in_lines = []
    start_idx = 0
    for line_no, idx in enumerate(split_indices):
        wrong_code_in_lines.append(
            {
                "line_no": line_no,
                "start_idx": start_idx,
                "end_idx": idx + 1,
                "tensor": wrong_code[:, start_idx:idx + 1]
            }
        )
        start_idx = idx + 1
    
    if start_idx < wrong_code.shape[-1]:
        wrong_code_in_lines.append(
            {
                "line_no": len(split_indices),
                "start_idx": start_idx,
                "end_idx": wrong_code.shape[-1],
                "tensor": wrong_code[:, start_idx:]
            }
        )


    # TODO: some signals
    START = 0
    GENERATE_WHOLE_LINE = 1
    GENERATE_NEXT_LINE_PREFIX = 2
    PADDING = 3
    AGGRESSIVE_GEN = 4
    PADDING_PREFIX_LENGTH = 5
    new_line_tokens = []
    accepted_length = input_ids.shape[-1]
    input_ids = torch.cat([input_ids, wrong_code], dim=-1)
    if input_ids.shape[-1] > max_length:
        input_ids = input_ids[:, :max_length]
    status = START
    model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
        input_ids, pad_token_id, eos_token_id
    )
    padding_start_line = 0
    forward_cnt = 0
    
    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        forward_cnt += 1
        '''should reuse these following code, use loop to check each token'''
        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        # next_token_logits = outputs.logits[:, -1, :]
        # TODO: this need to fix, outputs.logits is same as model_inputs.input_ids, without counting kvcache.
        
        if status == START:
            generated_token_logits = outputs.logits[:, accepted_length - 1:, :]
            status = PADDING
        else:
            generated_token_logits = outputs.logits


        # pre-process distribution
        # next_tokens_scores = logits_processor(input_ids, next_token_logits)
        generated_tokens_scores = logits_processor(input_ids, generated_token_logits)

        # Store scores, attentions and hidden_states when required
        '''return_dict_in_generate is False'''
        # if return_dict_in_generate:
        #     if output_scores:
        #         scores += (next_tokens_scores,)
        #     if output_attentions:
        #         decoder_attentions += (
        #             (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
        #         )
        #         if self.config.is_encoder_decoder:
        #             cross_attentions += (outputs.cross_attentions,)

        #     if output_hidden_states:
        #         decoder_hidden_states += (
        #             (outputs.decoder_hidden_states,)
        #             if self.config.is_encoder_decoder
        #             else (outputs.hidden_states,)
        #         )

        # argmax
        # next_tokens = torch.argmax(next_tokens_scores, dim=-1)
        generated_tokens = torch.argmax(generated_tokens_scores, dim=-1)

        # check generated tokens
        padded_correctly_length = 0
        # TODO: update padding start line
        if status == PADDING:
            padded_tokens = input_ids[:, accepted_length:]
            if not padded_tokens.shape[1] == 0:
                for i in range(padded_tokens.shape[1]):
                    if padded_tokens[0][i] == generated_tokens[0][i]:
                        padded_correctly_length = i + 1
                    else:
                        break

            padded_len = 0
            if padded_correctly_length > 0:
                for line in wrong_code_in_lines[padding_start_line:]:
                    if line["tensor"].shape[1] + padded_len >= padded_correctly_length:
                        padding_start_line = line["line_no"] + 1
                        break
                    padded_len += line["tensor"].shape[1]
            status = GENERATE_WHOLE_LINE
        
        # update generated ids, model inputs, and length for next step
        
        
        '''this is for padding eos to finished sentences, since we use batch_size = 1, so we can ignore this part'''
        # finished sentences should have their next token be a padding token
        # if eos_token_id is not None:
        #     if pad_token_id is None:
        #         raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
        #     next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
        '''part end'''
        # update generated ids, model inputs, and length for next step
        # input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        input_ids = torch.cat([input_ids[:, :accepted_length], generated_tokens[:, : padded_correctly_length + 1]], dim=-1)
        if status == GENERATE_NEXT_LINE_PREFIX:
            new_line_tokens.append(generated_tokens[0][padded_correctly_length])
            if new_line_tokens[-1] == 13:
                new_line_tokens = []
            if len(new_line_tokens) == PADDING_PREFIX_LENGTH:
                status = PADDING
        elif status == GENERATE_WHOLE_LINE:
            if generated_tokens[0][padded_correctly_length] == 13:
                status = GENERATE_NEXT_LINE_PREFIX
                new_line_tokens = []

        if status == PADDING:
            start_idx = -1
            if padding_start_line >= len(wrong_code_in_lines):
                status = AGGRESSIVE_GEN
                new_line_tokens = []
            else:
                for line in wrong_code_in_lines[padding_start_line:]:
                    if line["tensor"].shape[1] > 0:
                        # check whether the tensor start with new_line_tokens
                        if line["tensor"][0][0] == new_line_tokens[0]:
                            if line["tensor"].shape[1] >= len(new_line_tokens):
                                if (line["tensor"][0][:len(new_line_tokens)] == torch.tensor(new_line_tokens).cuda()).all().item():
                                    start_idx = line["start_idx"]
                                    break
                        else:
                            continue
                if start_idx == -1:
                    status = GENERATE_WHOLE_LINE
                    new_line_tokens = []
                else:
                    input_ids = torch.cat([input_ids, wrong_code[:, start_idx + PADDING_PREFIX_LENGTH:]], dim=-1)
                    if input_ids.shape[-1] > max_length:
                        input_ids = input_ids[:, :max_length]
                    new_line_tokens = []


        accepted_length = accepted_length + padded_correctly_length + 1
        if streamer is not None:
            # streamer.put(next_tokens.cpu())
            streamer.put(generated_tokens.cpu())
        # TODO: attention mask is not done.
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        # TODO: deal with padding or something else
        model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
            input_ids, pad_token_id, eos_token_id
        )
        # kvcache shape is dim_1, dim_2, (b, num_heads, seq_length, head_dim)
        if "past_key_values" in outputs:
            model_kwargs["past_key_values"] = tuple((
                tensor1[:, :, :accepted_length - 1, :], tensor2[:, :, :accepted_length - 1, :]
            ) for tensor1, tensor2 in model_kwargs["past_key_values"])    
        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            # unfinished_sequences = unfinished_sequences.mul(
            #     next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            # )
            unfinished_sequences = unfinished_sequences.mul(
                (~((generated_tokens[:, :padded_correctly_length + 1] == eos_token_id_tensor).any(dim=1))).long()
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GreedySearchEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return GreedySearchDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        print(forward_cnt)
        return input_ids, forward_cnt
    

def prepare_inputs_for_generation(
    self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    if past_key_values:
        # TODO: need to check
        # input_ids = input_ids[:, -1:]
        input_ids = input_ids[:, past_key_values[0][0].shape[2]:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            # position_ids = position_ids[:, -1:]
            position_ids = position_ids[:, past_key_values[0][0].shape[2]:].unsqueeze(-1)

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs