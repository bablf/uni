import tensorflow as tf
# pip install git+https://github.com/huggingface/transformers.git
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_tf_utils import TFPreTrainedModel
import re
import os
import numpy as np

from typing import List

class ProbesToFile:
    tokenizer:PreTrainedTokenizerBase
    model:TFPreTrainedModel

    def __init__(self, name:str = 'gpt2'):
        self.reset()
        # options are (from https://huggingface.co/transformers/pretrained_models.html)
        # 'gpt2' : 12-layer, 768-hidden, 12-heads, 117M parameters. # OpenAI GPT-2 English model
        # 'gpt2-medium' : 24-layer, 1024-hidden, 16-heads, 345M parameters. # OpenAI’s Medium-sized GPT-2 English model
        # 'gpt2-large' : 36-layer, 1280-hidden, 20-heads, 774M parameters. # OpenAI’s Large-sized GPT-2 English model
        # 'gpt2-xl' : 48-layer, 1600-hidden, 25-heads, 1558M parameters.. # OpenAI’s XL-sized GPT-2 English model
        self.tokenizer = GPT2Tokenizer.from_pretrained(name)

        # add the EOS token as PAD token to avoid warnings
        self.model = TFGPT2LMHeadModel.from_pretrained(name, pad_token_id=self.tokenizer.eos_token_id, from_pt=True)
        # model = TFGPT2LMHeadModel.from_pretrained("../data/moby_dick_model", pad_token_id=tokenizer.eos_token_id, from_pt=True)


    def reset(self):
            print("ProbesToFile.reset")

    def run_probe(self, probe:str, batch_size:int, batches:int=1, max_length:int=500) -> List[str]:
        to_return = []
        # encode context the generation is conditioned on
        input_ids = self.tokenizer.encode(probe, return_tensors='tf')
        print("probe = {}:".format(probe))
        for i in range(batches):
            print("batch {}".format(i))
            # generate text until the output length (which includes the context length) reaches 50
            output_list  = self.model.generate(
                input_ids,
                do_sample=True,
                max_length=max_length,
                top_k=50,
                top_p=0.95,
                num_return_sequences=batch_size)

            for i, beam_output  in enumerate(output_list):
                output = self.tokenizer.decode(beam_output , skip_special_tokens=True)
                text = " ".join(output.split())
                text = text[len(probe):]
                to_return.append(text)
                print("\t[{}]: {}".format(i, text))
        return to_return

    def get_valid_filename(self, s):
        s = str(s).strip().replace(' ', '_')
        return re.sub(r'(?u)[^-\w.]', '', s)

    def to_list(self, dict_list:List, key:str) -> List:
        to_return = []
        for d in dict_list:
            if d[key] != None:
                to_return.append(d[key])
        return to_return

    def to_file(self, filename:str, text_list:List[str]):
        with open(filename, "w", encoding="utf-8") as f:
            for t in text_list:
                f.write("{}\n\n".format(t))

    def to_print(self, text_list):
        for i in range(len(text_list)):
            t = text_list[i]
            print("\t[{}]: {}".format(i, t))

def main():
    tf.random.set_seed(2)
    os.chdir("../models")
    print(os.getcwd())
    probe_list = ["What the new movement prophesied again and again before those great masses of people has been fulfilled almost in every detail. "]
    for probe in probe_list:
        ptf = ProbesToFile("./gpt2-large")
        text_list = ptf.run_probe(probe, 1, 1)
        ptf.to_file("../results/{}.txt".format(ptf.get_valid_filename(probe)), text_list)

if __name__ == "__main__":
    main()