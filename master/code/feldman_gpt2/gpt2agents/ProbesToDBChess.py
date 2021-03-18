import os
import re
import json
from typing import List
from gpt2agents import ProbesToDBBase as PTdbB
from gpt2agents.utils.MySqlInterface import MySqlInterface as MSI

def main():
    os.chdir("../models")
    print(os.getcwd())
    msi = MSI("root", "postgres", "gpt_experiments")
    description = '''chess_model ecco trend test'''
    model_name = "./chess_model"
    batch_size = 5
    do_sample = True
    max_length = 20
    top_k = 50
    top_p = .95
    num_return_sequences = 5
    use_ecco = True
    test_tokens = False


    probe_list = ['The game begins as', 'In move 10', 'In move 20', 'In move 30', 'In move 40', 'White takes black', 'Black takes white']

    token_str = " pawn rook knight bishop queen king"

    print(probe_list)

    ptf = PTdbB.ProbesToDBBase(name=model_name, use_ecco=use_ecco, from_pt=True)
    if use_ecco and test_tokens:
        token_dict = ptf.eta.tokenize(token_str)
        print(token_dict)
        return

    split_str = "]]"
    regex_dict_list = [{ptf.LABEL:"location", ptf.BEFORE:"]]\s\w+\s\w+,\s", ptf.AFTER:"\s\[\["},
                       {ptf.LABEL:"tweet", ptf.BEFORE:"\[\[", ptf.AFTER:""},
                       {ptf.LABEL:ptf.DATE, ptf.BEFORE:"]] \w+ \d+,", ptf.AFTER:"]] %B %Y,"}]
    experiment_id = ptf.record_experiment(msi, description=description, model_name=model_name, probe_list=probe_list,
                                          batch_size=batch_size, do_sample=do_sample, max_length=max_length, top_k=top_k,
                                          top_p=top_p, num_return_sequences=num_return_sequences)

    if use_ecco:

        for probe in probe_list:
            dl = ptf.run_ecco_probe(probe, token_str, batch_size=batch_size, max_length=max_length, plot=False, verbose=True)
            # print(dl)
            root_id = ptf.record_row(msi, experiment_id=experiment_id, root_id=0, tag=ptf.RAW, depth=0, probe=probe, content="JSON", before_regex="")
            for d in dl:
                js = json.dumps(d)
                ptf.record_row(msi, experiment_id=experiment_id, root_id=root_id, tag=ptf.JSON, depth=1, probe=probe, content=js, before_regex="")

    else:
        for probe in probe_list:
            text_list = ptf.run_probe(probe, batch_size=batch_size, do_sample=do_sample, max_length=max_length, top_k=top_k,
                                      top_p=top_p, num_return_sequences=num_return_sequences)
            for tl in text_list:
                root_id = ptf.record_row(msi, experiment_id=experiment_id, root_id=0, tag=ptf.RAW,
                                         depth=0, probe=probe, content=tl, before_regex="")
                entry_list = ptf.parse_substrings(split_str=split_str, regex_dict_list=regex_dict_list, text_str=tl)

                tags = [ptf.FULL, ptf.TRIMMED]
                for tag in tags:
                    l = ptf.to_list(entry_list, tag)
                    ptf.record_list(msi, experiment_id=experiment_id, root_id=root_id, probe=probe, tag=tag, before_regex=split_str, after_regex="", content_list=l)

                for regex_dict in regex_dict_list:
                    tag = regex_dict[ptf.LABEL]
                    l = ptf.to_list(entry_list, tag)
                    before_regex = regex_dict[ptf.BEFORE]
                    after_regex = regex_dict[ptf.AFTER]
                    ptf.record_list(msi, experiment_id=experiment_id, root_id=root_id, probe=probe, tag=tag,
                                    before_regex=before_regex, after_regex=after_regex, content_list=l)

if __name__ == "__main__":
    main()