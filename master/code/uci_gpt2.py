"""
Autor: Florian Babl
Thema: Probing World Knowledge of Transformer Language Models: A Case Study on Chess

Muss ich selber implementieren und finetunen.
Get uci data from create_uci_dataset
Standard tokenizer for GPT2
GPT Version: GPT2Model or GPT2LMHeadModel

"""
from transformers import GPT2Model, GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import gpt_2_simple as gpt2
from gpt_2_simple.gpt_2 import download_gpt2
from gpt_2_simple.gpt_2 import encode_plain_dataset, encode_csv
import os

model_name = '355M'
if not os.path.isdir(os.path.join("models", model_name)):
    print(f"Downloading {model_name} model...")
    download_gpt2(model_name=model_name)  # model is saved into current directory under /models/124M/

sess = gpt2.start_tf_sess()
path_to_txt_file = "data/uci_dataset_with_tags.txt"
out_path = "data/uci_dataset_with_tags.npz"

# TODO wahr unnötig. colab konnte einfach die txt file nehmen
encode_plain_dataset(path_to_txt_file,
                     out_path=out_path,
                     model_name=model_name)
exit()
gpt2.finetune(sess,
              out_path,
              model_name=model_name,
              steps=30000)  # steps is max number of training steps

gpt2.generate(sess, run_name='run1', checkpoint_dir='checkpoint2', model_name=model_name,
              prefix='[Result "1-0"] e2e4',  # assigns the prefix of the already played game, gets encoded automatically
              length=10)  # assigns the length of the prediction?




if __name__ == "__main__":
    pass
