import re
from typing import List, Dict, Pattern

# pip install git+https://github.com/huggingface/transformers.git
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

import gpt2agents.utils.MySqlInterface as MSI

class ChessMovesToDb:
    model:TFGPT2LMHeadModel
    tokenizer:GPT2Tokenizer
    msi: MSI.MySqlInterface
    probe_list:List[str]
    num_probes: int
    batch_size: int
    square_regex: Pattern
    move_num_regex: Pattern
    model_dir:str
    table_name:str


    def __init__(self, probe_list:List[str], num_probes:int, batch_size:int, model_dir:str = "../data/chess_model", table_name:str = "table_moves", clear_table:bool = True):
        self.model_dir = model_dir
        self.table_name = table_name
        self.reset()
        self.probe_list = probe_list
        self.num_probes = num_probes
        self.batch_size = batch_size
        if clear_table:
            self.msi.write_data("truncate {}".format(table_name))

    def reset(self):
        print("ChessMovesToDb.reset()")
        self.msi = MSI.MySqlInterface("root", "postgres", "gpt2_chess")
        self.square_regex = re.compile('[a-h][1-8]')
        self.move_num_regex = re.compile('move \d*')
        self.probe_list = []
        self.num_probes = 1
        self.batch_size = 1
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = TFGPT2LMHeadModel.from_pretrained(self.model_dir,
            pad_token_id=self.tokenizer.eos_token_id, from_pt=True)


    #parameters based on this article: https://huggingface.co/blog/how-to-generate
    def evaluate(self, do_sample:bool=True, max_length:int=60, top_k:int=50, top_p:float=0.95):
        for probe in self.probe_list:
            input_ids = self.tokenizer.encode(probe, return_tensors='tf')

            # generate text until the output length (which includes the context length) reaches 50
            for b in range(self.batch_size):
                print("probe '{}', batch {}".format(probe, b))
                output_list  = self.model.generate(
                    input_ids,
                    do_sample=do_sample,
                    max_length=max_length,
                    top_k=top_k,
                    top_p=top_p,
                    num_return_sequences=self.num_probes)
                for i, beam_output  in enumerate(output_list):
                    output = self.tokenizer.decode(beam_output , skip_special_tokens=True)
                    m = " ".join(output.split())
                    moves = self.parse_move(probe, m)
                    for m in moves: # if moves is empty, we'll skip this
                        self.store_move(m)

    def store_move(self, d:Dict):
        r_str = self.msi.escape_text(d["response"])
        r_str = r_str.replace("'", "")
        sql_str = 'insert into {}(`move_number`, `color`, `piece`, `from`, `to`, `probe`, `response`) values ({}, "{}", "{}", "{}", "{}", "{}", "{}")'.format(
            self.table_name, d["move_number"], d["color"], d["piece"], d["from"], d["to"], d["probe"], r_str)
        print(sql_str)
        self.msi.write_data(sql_str)

    def get_move_num(self, s:str, marker:str = "move ") ->int:
        move_num = -1
        move_num_list = self.move_num_regex.findall(s)
        if len(move_num_list) > 0:
            s = move_num_list[0]
            result = s.partition(marker)[2]
            try:
                move_num = int(result)
            except ValueError:
                pass
        return move_num

    # may have to be parse_opening_move??
    def parse_move(self, probe_str:str, move_str:str, split_term:str="moves") -> List[Dict]:
        raw_str = move_str[len(probe_str):] # nope! Need to strip to "moves"

        move_num = self.get_move_num(probe_str)
        if move_num == -1:
            move_num = self.get_move_num(raw_str) # because this is probably picked up from the subsequent move
        move_num = max(1, move_num)

        print("\nprobe = {}, raw = {}".format(probe_str, move_str))
        piece_list = ['pawn', 'rook', 'knight', 'bishop', 'king', 'queen']
        move_str = "unset"
        for p in piece_list:
            s = "{} {}".format(split_term, p)
            if s in raw_str:
                move_str = raw_str.partition(split_term)[2]
                break

        if move_str != "unset":
            word_list = move_str.split()
            plist = []
            for w in word_list:
                if w in piece_list:
                    plist.append(w)
            square_list = self.square_regex.findall(move_str)
            # print("white moves {} from {} to {}".format(plist[0], square_list[0], square_list[1]))
            # print("black moves {} from {} to {}".format(plist[1], square_list[2], square_list[3]))
            try:
                w = {"move_number": move_num, "color":"white", "piece":plist[0], "from":square_list[0], "to":square_list[1], "probe":probe_str, "response":raw_str}
                b = {"move_number": move_num, "color":"black", "piece":plist[1], "from":square_list[2], "to":square_list[3], "probe":probe_str, "response":raw_str}
                print(w)
                print(b)
                return [w, b]
            except IndexError:
                print("parse_move() IndexError: plist = {}, square_list = {}".format(plist, square_list))


        return []



    def close(self):
        self.msi.close()

def main():
    probe_list = ['The game begins as ', 'In move 10', 'In move 20', 'In move 30', 'In move 40', 'White takes black ', 'Black takes white ', 'Check. ']
    cm = ChessMovesToDb(probe_list, 100, 100, model_dir="../data/chess_model_200", table_name="table_moves_200", clear_table=True)
    cm.evaluate()
    cm.close()

if __name__ == "__main__":
    main()