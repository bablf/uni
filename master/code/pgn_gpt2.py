"""
Autor: Florian Babl
Thema: Probing World Knowledge of Transformer Language Models: A Case Study on Chess

Model from Noever aka Chess transfomers.

Todo:
    - What was the medium model trained on? Which parameters were used?
    Trained on milibrary and kingsbase dataset. 30k steps and gpt-2-simple
"""
# folder checkpoint contains the fine-tuned model from Noever
# This is the Small Model GPT-2 (~350M parameter)

#from gpt_2_simple.src import encoder
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model, GPT2TokenizerFast, GPT2Tokenizer, PreTrainedTokenizer

# sess = gpt2.start_tf_sess()
# gpt2.generate(sess)
# gpt2.load_gpt2(sess)
# enc = encoder.get_encoder("checkpoint/run1/")

config = GPT2Config.from_json_file("noever_gpt2_checkpoint_huggingface_compatible/config.json")
model = GPT2Model.from_pretrained("pgn_checkpoint/run1/model-1000.index", from_tf=True, config=config)
tokenizer = GPT2Tokenizer("pgn_checkpoint/run1/encoder.json", "pgn_checkpoint/run1/vocab.bpe")  # the regular gpt2 tokenizer
print(config)
if __name__ == "__main__":
    pass