"""
Autor: Florian Babl
Thema: Probing World Knowledge of Transformer Language Models: A Case Study on Chess

use the one from Toshniwal on huggingface
"""

from transformers import GPT2LMHeadModel
# TODO: import ChessTokenizer from learning-chess-blindfolded @shtoshni
import sys
import os
from learning_chess_blindfolded.src.data_utils.chess_tokenizer import ChessTokenizer


model = GPT2LMHeadModel.from_pretrained("shtoshni/gpt2-chess-uci")
# TODO: Einmal downloaden und abspeichern
tokenizer = ChessTokenizer()


if __name__ == "__main__":
    pass