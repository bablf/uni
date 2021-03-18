import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_tf_utils import TFPreTrainedModel
import matplotlib.pyplot as plt
from datetime import datetime
import re
import os
from gpt2agents.utils.MySqlInterface import MySqlInterface as MSI
from gpt2agents.analytics import EccoTrendAnalytics as ETA
import pymysql
from typing import List, Dict

class ProbesToDBBase:
    tokenizer:PreTrainedTokenizerBase
    model:TFPreTrainedModel
    eta:ETA
    RAW = "raw"
    FULL = "full"
    TRIMMED = "trimmed"
    LABEL = "label"
    BEFORE = "before"
    AFTER = "after"
    DATE = "date"
    JSON = "json"

    def __init__(self, seed:int = 1, name:str = 'gpt2', use_ecco:bool=False, from_pt:bool = False):
        self.reset()
        tf.random.set_seed(seed)
        # options are (from https://huggingface.co/transformers/pretrained_models.html)
        # 'gpt2' : 12-layer, 768-hidden, 12-heads, 117M parameters. # OpenAI GPT-2 English model
        # 'gpt2-medium' : 24-layer, 1024-hidden, 16-heads, 345M parameters. # OpenAI’s Medium-sized GPT-2 English model
        # 'gpt2-large' : 36-layer, 1280-hidden, 20-heads, 774M parameters. # OpenAI’s Large-sized GPT-2 English model
        # 'gpt2-xl' : 48-layer, 1600-hidden, 25-heads, 1558M parameters.. # OpenAI’s XL-sized GPT-2 English model

        if use_ecco:
            self.eta = ETA.EccoTrendAnalytics(name)
        else:
            self.tokenizer = GPT2Tokenizer.from_pretrained(name)

            # add the EOS token as PAD token to avoid warnings
            self.model = TFGPT2LMHeadModel.from_pretrained(name, pad_token_id=self.tokenizer.eos_token_id, from_pt=from_pt)

    def reset(self):
        print("ProbesToDBBase.reset")
        self.eta = None

    '''
    batch_size = 1
    do_sample = True
    max_length = 1000
    top_k = 50
    top_p = .95
    '''
    def record_experiment(self, msi:MSI, description:str, model_name:str, probe_list:List, batch_size:int=1, do_sample:bool=True, max_length:int = 500, top_k:int=50, top_p:float = 0.95, num_return_sequences:int=5, debug=False) -> int:
        now = datetime.now()
        probes = "[{}]".format(probe_list[0])
        for i in range(1, len(probe_list)):
            probes += ", [{}]".format(probe_list[i])
        sql = '''insert into table_experiment 
            (description, model_name, date, probe_list, batch_size, do_sample, max_length, top_k, top_p, num_return_sequences) 
            values ({}, {}, {}, {}, {}, {}, {}, {}, {}, {});'''.format(
            msi.escape_text(description), msi.escape_text(model_name), msi.escape_text(now.strftime('%Y-%m-%d %H:%M:%S')),
            msi.escape_text(probes), batch_size, do_sample, max_length, top_k, top_p, num_return_sequences)
        print(sql)
        if debug:
            return 0
        try:
            id = msi.write_data_get_row_id(sql)
            print("row id = {}".format(id))
            return id
        except pymysql.err.InternalError:
            pass

    def record_row(self, msi:MSI, experiment_id:int, root_id:int, tag:str, depth:int, probe:str, content:str, before_regex:str, after_regex:str="", debug:bool = False) -> int:
        sql = '''insert into table_output (experiment_id, root_id, tag, depth, probe, content, before_regex, after_regex) VALUES 
        ({}, {}, {}, {}, {}, {}, {}, {})'''.format(experiment_id, root_id, msi.escape_text(tag),
                                               depth, msi.escape_text(probe), msi.escape_text(content),
                                                    msi.escape_text(before_regex), msi.escape_text(after_regex))
        print(sql)
        if debug:
            return 0
        try:
            id = msi.write_data_get_row_id(sql)
            return id
        except pymysql.err.InternalError:
            pass


    def record_list(self, msi:MSI, experiment_id:int, root_id:int, probe:str, tag:str, before_regex:str, after_regex:str, content_list:List, debug:bool = False):
        for i in range(len(content_list)):
            content = content_list[i]
            depth = i+1
            self.record_row(msi, experiment_id=experiment_id, root_id=root_id, tag=tag, depth=depth,
                           probe=probe, content=content, before_regex=before_regex, after_regex=after_regex, debug=debug)

    def run_ecco_probe(self, probe:str, token_str:str, batch_size:int=5, max_length:int = 20, layer_num:int = 0,
                       plot:bool = False, verbose:bool = False) -> List[Dict]:
        to_return = []
        if self.eta == None:
            print("run_ecco_probe() WARNING! Running without ecco instance. Use run_probe instead")
            return to_return
        print("probe = {}:".format(probe))
        for i in range(batch_size):
            print("batch {}".format(i))
            d = self.eta.token_ranks(probe, token_str, plot=plot, tokens_to_generate=max_length, layer_num=layer_num,
                                     verbose=verbose)
            if plot:
                plt.show()
            to_return.append(d)
        return to_return

    def run_probe(self, probe:str, batch_size:int=1, do_sample:bool=True, max_length:int = 500, top_k:int=50, top_p:float = 0.95, num_return_sequences:int=5) -> List[str]:
        to_return = []
        if self.eta != None:
            print("run_probe() WARNING! Running using Ecco. Use run_ecco_probe instead")
            return to_return

        # encode context the generation is conditioned on
        input_ids = self.tokenizer.encode(probe, return_tensors='tf')
        print("probe = {}:".format(probe))
        for i in range(batch_size):
            print("batch {}".format(i))
            # generate text until the output length (which includes the context length) reaches 50
            output_list  = self.model.generate(
                input_ids,
                do_sample=do_sample,
                max_length=max_length,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences)

            for i, beam_output  in enumerate(output_list):
                output = self.tokenizer.decode(beam_output , skip_special_tokens=True)
                text = " ".join(output.split())
                to_return.append(text)
                #print("\t[{}]: {}".format(i, text))
        return to_return

    def get_substring_between(self, before_str:str, after_str:str, test_str:str) -> str:
        before_regex = re.compile(r"({})".format(before_str))
        before_match = before_regex.search(test_str)
        if before_match == None:
            return None
        print("\tbefore_match start = {}, end = {}".format(before_match.start(), before_match.end()))
        if after_str == "":
            offset = before_match.end()
            return test_str[offset:]

        after_regex = re.compile(r"({})+".format(after_str))
        after_match_iter = after_regex.finditer(test_str)
        for am in after_match_iter:
            # print("\tafter_match start = {}, end = {}".format(am.start(), am.end()))
            if am.start() > before_match.end():
                return test_str[before_match.end():am.start()]

        return None

    ''' Parse out the substrings based on the passed-in regex. This allows the 
        probe to be different from the split that the model has been trained to repeat.
        This turns out to be slightly trickey, because the matches don't include the 
        text we're interested in, so we have a step that splits out all the individual
        texts. We also throw away the text that leads to the first line and the last line
        of text since both can be assumed to be incomplete'''
    def parse_substrings(self, split_str:str, regex_dict_list:List, text_str:str) -> List:
        text_str = " ".join(text_str.split('\n'))
        split_regex = re.compile(r"({})".format(split_str))
        split_iter = split_regex.finditer(text_str)
        start = 0
        text_list = []
        for s in split_iter:
            t = text_str[start: s.start()]
            text_list.append(t)
            start = s.start()
        to_return = []
        if len(text_list) > 0:
            entry = {self.TRIMMED:text_list[0]}
            to_return.append(entry)
        for i in range(1, len(text_list)):
            entry = {}
            full_str = text_list[i]
            entry[self.FULL] = full_str
            for regex_dict in regex_dict_list:
                label = regex_dict[self.LABEL]
                if label == self.DATE:
                    date_regex = re.compile(regex_dict[self.BEFORE])
                    results = date_regex.findall(full_str)
                    try:
                        s = results[0]
                        f = regex_dict[self.AFTER]
                        d = datetime.strptime(s, f)
                        entry[label] = d.strftime("%Y-%m-%d")
                    except IndexError as e:
                        print("ProbesToDBBase.parse_substrings()IndexError: {}".format(e))
                    except ValueError as e:
                        print("ProbesToDBBase.parse_substrings() ValueError: {}".format(e))
                else:
                    entry[label] = self.get_substring_between(regex_dict[self.BEFORE], regex_dict[self.AFTER], full_str)
            to_return.append(entry)
            # print("[{}]: {}".format(i, full_str))

        return to_return

    def get_valid_filename(self, s):
        s = str(s).strip().replace(' ', '_')
        return re.sub(r'(?u)[^-\w.]', '', s)

    def to_list(self, dict_list:List, key:str) -> List:
        to_return = []
        for d in dict_list:
            if key in d:
                to_return.append(d[key])
        return to_return

    def to_print(self, text_list):
        for i in range(len(text_list)):
            t = text_list[i]
            print("\t[{}]: {}".format(i, t))

def main():
    os.chdir("../models")
    print(os.getcwd())

    msi = MSI("root", "postgres", "gpt_experiments")
    description = '''Test to see if everything works'''
    model_name = "./GPT-2_small_English_Twitter"
    probe_list = ["On March of 2020, ", "On April of 2020, ", "On May of 2020, "]
    batch_size = 1
    do_sample = True
    max_length = 400
    top_k = 50
    top_p = .95
    num_return_sequences = 5
    ptf = ProbesToDBBase(name="./GPT-2_small_English_Twitter", from_pt=True)
    experiment_id = ptf.record_experiment(msi, description=description, model_name=model_name, probe_list=probe_list,
                          batch_size=batch_size, do_sample=do_sample, max_length=max_length, top_k=top_k,
                          top_p=top_p, num_return_sequences=num_return_sequences)

    component_list = [ptf.FULL]
    split_str = "(On [a-zA-Z]+ of [0-9]+,)"
    regex_dict_list = [{"label":"Location", "before":"posted a tweet from ", "after":"\. "},
                       {"label":"tweet", "before":"They were using [a-zA-Z0-9\., ]+\. \"", "after":"\" "}]
    for probe in probe_list:
        text_list = ptf.run_probe(probe, batch_size=batch_size, do_sample=do_sample, max_length=max_length, top_k=top_k,
                                  top_p=top_p, num_return_sequences=num_return_sequences)
        for tl in text_list:
            #ptf.to_print(text_list)
            root_id = ptf.record_row(msi, experiment_id=experiment_id, root_id=0, tag=ptf.RAW,
                                     depth=0, probe=probe, content=tl, before_regex="")
            entry_list = ptf.parse_substrings(split_str=split_str, regex_dict_list=[], text_str=tl)
            for c in component_list:
                l = ptf.to_list(entry_list, c)
                ptf.record_list(msi, experiment_id=experiment_id, root_id=root_id, probe=probe, tag=c, before_regex=split_str, content_list=l)

if __name__ == "__main__":
    main()