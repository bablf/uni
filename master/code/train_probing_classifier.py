"""
Autor: Florian Babl
Thema: Probing Classifiers for LM in a Chess setting
"""
from abc import ABC
from typing import Iterator

from torch.utils.data.dataset import T_co

""" GPT-2 MODELLE """
# Auf welche Modelle habe ich zugriff?
#   1. Auf gpt2 mit UCI & PGN notation finetuned
#   2. Auf gpt2 mit converted PGN finetuned
#   3. pretrained gpt2 (kann einfach aus Huggingface importiert werden)

""" PROBING """
# TODO: Probing Classifier erstellen, und trainieren.
# 1. Neuronales Netz ausdenken für alle Probing Classifier
# Datensatz für probing erstellen
# Datensatz erstellen, der Überprüft werden soll.
# Aufbau von Datensatz: String von PGN UND UCI notation --> Schachfeld.
# beides, damit man sowohl PGN als auch UCI nutzen könnte.

import argparse
from convert_game import preprocess_game_string
import chess.pgn as pgn
from create_probing_dataset import create_dataset
from models import UciGPT
from torch.utils.data import TensorDataset, random_split
from probing_classifier import ProbingChess
import pickle
from transformers import pipeline
from transformers import modeling_utils
from torch import nn
from torch.utils.data import IterableDataset, Dataset, DataLoader
import torch
import numpy as np


class ChessGamesDataset(IterableDataset):  # todo maybe iterabledataset if memory problems
    def __init__(self, filename, numb_games, notation, tokenizer):
        self.filename = filename
        self.notation = notation
        self.tokenizer = tokenizer
        self.train_counter = round(numb_games * 0.8)
        self.dev_counter = round(numb_games * 0.1)
        # get longest tokenized sequence so padding is possible
        if model.notation == "uci":
            self.tokenizer.model_max_length = model.tokenizer(open("data/max_length_uci.txt").readline(),
                                                              return_length=True)['length']
        else:  # == pgn
            self.tokenizer.model_max_length = model.tokenizer(open("data/max_length_pgn.txt").readline(),
                                                              return_length=True)['length']

    def game_mapper(self, file_iter):
        game = file_iter.split(";")
        board = np.fromstring(game[-1][1:-1], dtype=int, sep=' ')
        # give uci or pgn to the tokenizer
        print(board)
        return self.tokenizer(game[0] if self.notation == "pgn" else game[1],
                              return_tensors='pt',
                              padding='max_length'), board

    def __iter__(self):
        # Create an iterator
        file_itr = open(self.filename, encoding='cp1252')
        # Map each element using the line_mapper
        mapped_itr = map(self.game_mapper, file_itr)
        return mapped_itr


def train_probing_classifier(chess_model, dataset, probing_classifier):
    c = 0
    for x, boards in DataLoader(dataset, batch_size=1):  # todo set batch
        print(boards)
        exit()
        if c > dataset.dev_counter:
            test()
        elif c > dataset.train_counter:
            dev()
        else: # train()
            train()

        print(x.input_ids)
        #pt_outputs = chess_model.model(**games)  # todo check  shape (batch_size, sequence_length, hidden_size)
        #probing_classifier(pt_outputs.last_hidden_state)

        #pt_outputs = chess_model.model(**games)  # todo check  shape (batch_size, sequence_length, hidden_size)
        probing_classifier(x.last_hidden_state)
        exit()

    # for game in dataset:
    #     enc, board = encode_game(chess_model, game)  # todo batchsize =16 ?
    #     pt_outputs = chess_model.model(**enc)  # todo check  shape (batch_size, sequence_length, hidden_size)
    #     print("Output:\n" + 100 * '-')         # todo: Vl sequence_lenght auf einheitliche (größte) Größe padden.
    #     print(pt_outputs.last_hidden_state.shape)  # FloatTensor of shape (batch_size, sequence_length, hidden_size)
    #     probing_classifier(pt_outputs.last_hidden_state)
    #     exit()

# print(games.input_ids.shape)
        # print()
        # print(boards.shape)
        # # enc = encode_game(chess_model, games)  # todo batchsize =16 ?
        # pt_outputs = chess_model.model(**enc)  # todo check  shape (batch_size, sequence_length, hidden_size)
        # print("Output:\n" + 100 * '-')         # todo: Vl sequence_lenght auf einheitliche (größte) Größe padden.
        # print(pt_outputs.last_hidden_state.shape)  # FloatTensor of shape (batch_size, sequence_length, hidden_size)
        # #FloatTensor of shape (batch_size, sequence_length, hidden_size)
        # #                           16   ,      512       ,
        # exit()
        # ProbingChess(pt_outputs.hidden_states)
        # todo reset_session?

if __name__ == "__main__":
    chess_models = [UciGPT]
    for model in chess_models:  # ["pretrained_gpt","pgn_gpt", "uci_gpt", "special_gpt"]
        # Creating the iterable dataset object
        dataset = ChessGamesDataset("data/probing_dataset.txt", model.notation, model.tokenizer)
        # Create Probing_classifier
        probing_classifier = ProbingChess(model.config.n_embd, dataset.tokenizer.model_max_length)
        # todo split()
        train_probing_classifier(model, dataset, probing_classifier)
        # train_probing_classifier(model, dataset, probing_classifier)
        # todo save
