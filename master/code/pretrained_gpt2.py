"""
Autor: Florian Babl
Thema: Probing World Knowledge of Transformer Language Models: A Case Study on Chess

use model from Toshniwal on huggingface (GPT2LMHeadModel)
"""

from transformers import GPT2LMHeadModel, GPT2Model
from learning_chess_blindfolded.src.data_utils.chess_tokenizer import ChessTokenizer

model = GPT2LMHeadModel.from_pretrained("shtoshni/gpt2-chess-uci")
vocab_file = "learning_chess_blindfolded/sample_data/lm_chess/vocab/uci/vocab.txt"
tokenizer = ChessTokenizer(vocab_file, notation='uci', pad_token="<pad>", bos_token="<s>", eos_token="</s>")
print(tokenizer.eos_token)


if __name__ == "__main__":
    pass