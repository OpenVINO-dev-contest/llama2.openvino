from transformers import AutoTokenizer
import openvino as ov
from pathlib import Path
import numpy as np
import argparse
import time
import sys
utils_file_path = Path('.')
sys.path.append(str(utils_file_path))
from utils.memory_profile import MemConsumption


def model_has_state(ov_model: ov.Model):
    # TODO: Provide a better way based on the variables availability, but OV Python API doesn't expose required methods
    return len(ov_model.get_sinks()) > 0


def sample_next_token(logits: np.ndarray, top_k=20, top_p=0.8, temperature=1):
    # softmax with temperature
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits / temperature)
    probs = exp_logits / np.sum(exp_logits)

    # top k
    top_k_idx = np.argsort(-probs)[:top_k]
    top_k_probs = probs[top_k_idx]

    # top p
    cumsum_probs = np.cumsum(top_k_probs)
    top_k_probs[(cumsum_probs - top_k_probs) > top_p] = 0.0
    top_k_probs = top_k_probs / np.sum(top_k_probs)

    # sample
    next_token = np.random.choice(top_k_idx, size=1, p=top_k_probs)
    return next_token[0].item()


class llama2():
    def __init__(self,
                 model_path,
                 device='CPU') -> None:

        ir_model_path = Path(model_path)
        ir_model_file = ir_model_path / "openvino_model.xml"

        print(" --- loading tokenizer --- ")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        core = ov.Core()

        print(" --- reading model --- ")
        # read the model and corresponding weights from file
        self.model = core.read_model(ir_model_file)
        self.model_has_sinks = model_has_state(self.model)
        # input & output names
        self.input_names = {
            key.get_any_name(): idx
            for idx, key in enumerate(self.model.inputs)
        }
        self.output_names = {
            key.get_any_name(): idx
            for idx, key in enumerate(self.model.outputs)
        }
        self.key_value_input_names = [
            key for key in self.input_names if "key_values" in key
        ]
        self.key_value_output_names = [
            key for key in self.output_names if "present" in key
        ]
        print(" --- model compiling --- ")
        # compile the model for CPU devices
        self.request = core.compile_model(
            model=self.model, device_name=device).create_infer_request()
        self.eos_token_id = [self.tokenizer.eos_token_id]

    def generate_sequence(self,
                          input_ids,
                          max_generated_tokens=100,
                          top_k=50,
                          top_p=1,
                          temperature=0.1):
        attention_mask = np.ones((input_ids.shape[0], input_ids.shape[1]),
                                 dtype=np.int64)
        position_ids = np.arange(0, input_ids.shape[1], dtype=np.int64)
        position_ids = np.expand_dims(position_ids, axis=0)
        past_key_values = None
        num_iteration = 0
        other_latency = 0
        output_tokens = []
        new_position_id = np.copy(position_ids[..., -1:])
        inputs = {"position_ids": position_ids}
        self.request.reset_state()
        while True:
            inputs["input_ids"] = input_ids
            if not self.model_has_sinks:
                if past_key_values is not None:
                    inputs.update(past_key_values)
                else:
                    shape_input_ids = input_ids.shape
                    num_attention_heads = 1
                    for input_name in self.key_value_input_names:
                        model_inputs = self.model.input(input_name)
                        shape = model_inputs.get_partial_shape()
                        shape[0] = shape_input_ids[0] * num_attention_heads
                        if shape[2].is_dynamic:
                            shape[2] = 0
                        if shape[1].is_dynamic:
                            shape[1] = 0
                        inputs[input_name] = ov.Tensor(
                            model_inputs.get_element_type(), shape.get_shape())
            else:
                next_beam_idx = np.arange(1, dtype=int)
            if "attention_mask" in self.input_names and attention_mask is not None:
                inputs["attention_mask"] = attention_mask
            if "beam_idx" in self.input_names and next_beam_idx is not None:
                inputs["beam_idx"] = next_beam_idx
            before = time.perf_counter()
            self.request.start_async(inputs, share_inputs=True)
            self.request.wait()
            after = time.perf_counter()
            if num_iteration == 0:
                first_latency = after - before
            else:
                other_latency += after - before
            num_iteration += 1
            logits = self.request.get_tensor("logits").data
            if not self.model_has_sinks:
                past_key_values = tuple(
                    self.request.get_tensor(key).data
                    for key in self.key_value_output_names)
                past_key_values = {
                    k: v
                    for k, v in zip(self.key_value_input_names, past_key_values)
                }
            next_token = sample_next_token(logits[0, -1],
                                           top_k=top_k,
                                           top_p=top_p,
                                           temperature=temperature)

            output_tokens += [next_token]
            if next_token in self.eos_token_id or len(
                    output_tokens) > max_generated_tokens:
                break
            attention_mask = np.concatenate((attention_mask, [[1]]), axis=-1)
            new_position_id += 1
            inputs["position_ids"] = new_position_id
            input_ids = np.array([[next_token]], dtype=np.longlong)
        return output_tokens, num_iteration, (first_latency, other_latency)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h',
                        '--help',
                        action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-m',
                        '--model_path',
                        required=True,
                        type=str,
                        help='Required. path of IR model and tokenizer')
    parser.add_argument('-p',
                        '--prompt',
                        required=True,
                        type=str,
                        help='Required. prompt sentence')
    parser.add_argument('-l',
                        '--max_sequence_length',
                        default=128,
                        required=False,
                        type=int,
                        help='maximun lengh of output')
    parser.add_argument('-d',
                        '--device',
                        default='CPU',
                        required=False,
                        type=str,
                        help='device for inference')
    parser.add_argument('-s',
                        '--sampling',
                        default=True,
                        required=False,
                        type=bool,
                        help='sampling or not')
    args = parser.parse_args()

    mem_consumption = MemConsumption()
    ir_model_path = Path(args.model_path)

    max_rss_mem_consumption = ''
    max_shared_mem_consumption = ''
    mem_consumption.start_collect_mem_consumption_thread()
    mem_consumption.start_collect_memory_consumption()
    ov_model = llama2(ir_model_path, args.device)
    inputs = ov_model.tokenizer(args.prompt, return_tensors="np")
    input_len = len(inputs["input_ids"][0])

    print(" --- start generating --- ")
    start = time.perf_counter()
    response, num_tokens, latencies = ov_model.generate_sequence(
        inputs["input_ids"], max_generated_tokens=args.max_sequence_length)
    end = time.perf_counter()
    mem_consumption.end_collect_momory_consumption()
    max_rss_mem_consumption, max_shared_mem_consumption = mem_consumption.get_max_memory_consumption()
    mem_consumption.clear_max_memory_consumption()
    # Convert IDs to words and make the sentence from it

    print(" --- text decoding --- ")
    output_text = ov_model.tokenizer.decode(response, skip_special_tokens=True)
    print(f"Response: {output_text}")

    print(" --- Benchmarking --- ")
    print(f"Input length: {input_len} tokens")
    print(
        f"Generated {num_tokens} tokens in {end - start:.2f} s on {args.device}")
    print(
        f"Maximum rss memory consumption: {max_rss_mem_consumption:.2f} MB, Maximum shared memory consumption: {max_shared_mem_consumption:.2f}  MB")
    print(
        f"First inference latency: {1000*latencies[0]:.2f} ms/token, Other inference latency {1000*latencies[1]/(num_tokens-1):.2f} ms/token in average")
    mem_consumption.end_collect_mem_consumption_thread()
