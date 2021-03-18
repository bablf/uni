import re
import sys
import os
from typing import List, Dict, Pattern

from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

class GamesToSequences:
    model:TFGPT2LMHeadModel
    tokenizer:GPT2Tokenizer
    ending_list:List[str]
    probe:str
    num_sequences: int
    square_regex: Pattern
    filename:str

    def __init__(self, filename:str, num_sequences:int = 1,  clear:bool = True):
        self.reset()
        self.filename = filename
        self.num_sequences = num_sequences
        if clear and os.path.exists(self.filename):
            os.remove(self.filename)

    def reset(self):
        print("GamesToSequences.reset()")
        self.square_regex = re.compile('[a-h][1-8]')
        self.ending_list = ["wins", "resigns", "draw"]
        self.num_sequences = 1
        self.probe = 'The game begins as '
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = TFGPT2LMHeadModel.from_pretrained("../data/chess_model",
                                                       pad_token_id=self.tokenizer.eos_token_id, from_pt=True)

    def evaluate(self, do_sample:bool=True, max_length:int=1000, top_k:int=50, top_p:float=0.95):
        input_ids = self.tokenizer.encode(self.probe, return_tensors='tf')
        # generate text until the output length (which includes the context length) reaches 50
        for i in range(self.num_sequences):
            output_list  = self.model.generate(
                input_ids,
                do_sample=do_sample,
                max_length=max_length,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=10)
            for i, beam_output  in enumerate(output_list):
                output = self.tokenizer.decode(beam_output , skip_special_tokens=True)
                self.parse_sequence(output)

    def parse_sequence(self, text:str):
        end_pos = sys.maxsize
        for e in self.ending_list:
            pos = text.find(e)
            if pos > 0:
                text = text[:pos]

        square_list = self.square_regex.findall(text)
        text = " ".join(square_list)
        print("sequence = {}".format(text))
        with open(self.filename, "a") as f:
            f.write(text+"\n")


def main():
    gts = GamesToSequences("../data/chess/sequences.txt", 10)
    gts.evaluate()

if __name__ == "__main__":
    main()