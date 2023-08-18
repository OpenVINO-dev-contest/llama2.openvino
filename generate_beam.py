from transformers import LlamaTokenizer
from openvino.runtime import Core, Tensor
import numpy as np
import argparse
import time

beam_width = 4
num_hypotheses = 1

class Node(object):
    def __init__(self, parent, state, value, cost):
        super(Node, self).__init__()
        self.value = value
        self.parent = parent # parent Node, None for root
        self.state = state if state is not None else None # recurrent layer hidden state
        self.cum_cost = parent.cum_cost + cost if parent else cost # e.g. -log(p) of sequence up to current node (including)
        self.length = 1 if parent is None else parent.length + 1
        self._sequence = None
        
    def to_sequence(self):
        # Return sequence of nodes from root to current node.
        if not self._sequence:
            self._sequence = []
            current_node = self
            while current_node:
                self._sequence.insert(0, current_node)
                current_node = current_node.parent
        return self._sequence

    def to_sequence_of_values(self):
        return [[s.value[0][0] for s in self.to_sequence()]]



def process_logits(cur_length, scores, eos_token_id, min_length=0):
    if cur_length < min_length:
        scores[:, eos_token_id] = -float("inf")
    return scores


def generate_sequence(input_ids, attention_mask, eos_token_id,
                      max_sequence_length):
    shape_input_ids = input_ids.shape
    num_attention_heads = 1
    past_key_values = []
    for input_name in key_value_input_names:
        model_inputs = model.input(input_name)
        shape = model_inputs.get_partial_shape()
        shape[0] = shape_input_ids[0] * num_attention_heads
        if shape[2].is_dynamic:
            shape[2] = 0
        if shape[1].is_dynamic:
            shape[1] = 0
        past_key_values.append(Tensor(model_inputs.get_element_type(),
                                    shape.get_shape()))
        
    next_fringe = [Node(parent=None, state=past_key_values, value=input_ids, cost=0.0)]
    first = True
    count = 0
    hypotheses = []
    total_len = len(input_ids[0])
    for _ in range(max_sequence_length):

        fringe = []
        for n in next_fringe:
            if n.value[0][-1] == eos_token_id or total_len == max_sequence_length:
                hypotheses.append(n)
            else:
                fringe.append(n)

        if not fringe:
            break
        if first==False:
            past_key_values=[]
            for i in range(len(fringe[0].state)):
                past_key_values.append(np.concatenate([n.state[i] for n in fringe]))
            cur_input_len = 1
            total_len += 1 
            attention_mask = np.ones((len(fringe), total_len))
        else:
            cur_input_len = len(input_ids[0])
        first = False
        inputs = dict(zip(key_value_input_names,past_key_values)) 
        inputs["input_ids"] = np.concatenate([n.value for n in fringe])
        inputs["attention_mask"] = attention_mask
        
        request.start_async(inputs, shared_memory=True)
        request.wait()
        count += 1
        logits = request.get_tensor("logits").data
        present_key_values = [
            request.get_tensor(key).data for key in key_value_output_names]
        next_token_logits = logits[:, cur_input_len - 1, :]
        # pre-process distribution
        next_token_scores = process_logits(len(input_ids[0]),
                                           next_token_logits, eos_token_id)

        Y_t = np.argsort(next_token_scores, axis=1)[:,-beam_width:]
        i=0
        next_fringe = []
        for Y_t_n, p_t_n, n in zip(Y_t, next_token_scores, fringe):
            
            Y_nll_t_n = -np.log(p_t_n[Y_t_n])

            for y_t_n, y_nll_t_n in zip(Y_t_n, Y_nll_t_n):
                n_new = Node(parent=n,  state=[[state[i]] for state in present_key_values], value=[[y_t_n]], cost=y_nll_t_n)
                next_fringe.append(n_new)
            i += 1
        next_fringe = sorted(next_fringe, key=lambda n: n.cum_cost)[:beam_width] # may move this into loop to save memory
    hypotheses.sort(key=lambda n: n.cum_cost)
    return hypotheses[:num_hypotheses]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h',
                        '--help',
                        action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-m',
                        '--model_id',
                        default="meta-llama/Llama-2-7b-hf",
                        required=False,
                        type=str,
                        help='Required. hugging face model id')
    parser.add_argument('-p',
                        '--prompt',
                        required=False,
                        default="what is openvino ?",
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

    num_pkv = 2
    core = Core()

    print(" --- reading model --- ")
    # read the model and corresponding weights from file
    model = core.read_model('/home/ethan/intel/llama2.openvino/ir_model/openvino_model.xml')
    input_names = {
        key.get_any_name(): idx
        for idx, key in enumerate(model.inputs)
    }
    output_names = {
        key.get_any_name(): idx
        for idx, key in enumerate(model.outputs)
    }
    key_value_input_names = [key for key in input_names if "key_values" in key]
    key_value_output_names = [key for key in output_names if "present" in key]

    print(" --- model compiling --- ")
    # compile the model for CPU devices
    request = core.compile_model(
        model=model, device_name=args.device).create_infer_request()

    tokenizer = LlamaTokenizer.from_pretrained(args.model_id)
    inputs = tokenizer(args.prompt, return_tensors="np")

    print(" --- start generating --- ")
    start = time.perf_counter()
    output_ids = generate_sequence(
        inputs["input_ids"],
        inputs["attention_mask"],
        eos_token_id=tokenizer.eos_token_id,
        max_sequence_length=args.max_sequence_length,
    )
    end = time.perf_counter()
    output_text = " "
    # Convert IDs to words and make the sentence from it
    for output in output_ids:
        print(" --- text decoding --- ")
        output_id = np.array(output.to_sequence_of_values())
        output_text = tokenizer.batch_decode(output_id,
                                            skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)[0]
        print(f"Response: {output_text}")
