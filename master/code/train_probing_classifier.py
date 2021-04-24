"""
Autor: Florian Babl
Thema: Probing Classifiers for LM in a Chess setting


PROBING

# TODO: Probing Classifier erstellen, und trainieren.
# 1. Neuronales Netz ausdenken für alle Probing Classifier
# Datensatz für probing erstellen
# Datensatz erstellen, der Überprüft werden soll.
# Aufbau von Datensatz: String von PGN UND UCI notation --> Schachfeld.
# beides, damit man sowohl PGN als auch UCI nutzen könnte.
"""

import json

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch import from_numpy
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset, DataLoader

from models import PgnGPT
from probing_classifier import ProbingChess

seed_everything(42)


class ChessGamesDataset(IterableDataset):
    def __init__(self, filename, notation, tokenizer):
        self.filename = filename
        self.notation = notation
        self.tokenizer = tokenizer
        # get longest tokenized sequence so padding is possible
        if model.notation == "uci":
            self.tokenizer.model_max_length = model.tokenizer(open("data/max_length_uci.txt").readline(),
                                                              return_length=True)['length'][0] + 1
        else:  # == pgn
            self.tokenizer.model_max_length = model.tokenizer(open("data/max_length_pgn.txt").readline(),
                                                              return_length=True)['length'][0] + 1

    def game_mapper(self, file_iter):
        game = json.loads(file_iter)
        board = np.array(game["board"])
        indices = self.get_indices(game)
        # todo chess_blindfolded
        if self.notation == "pgn":
            self.tokenizer(game["pgn"])
        elif self.notation == "uci":
            self.tokenizer(game["uci"])
        elif self.notation == "uci-blindfolded":
            self.tokenizer(" ".join(game["uci"].split()[2:]))

        return self.tokenizer(game["pgn"] if self.notation == "pgn" else game["uci"], return_tensors='pt',
                              padding='max_length', return_offsets_mapping=True), from_numpy(board), indices

    def __iter__(self):
        # Create an iterator
        file_itr = open(self.filename, encoding='cp1252')
        # Map each element using the line_mapper
        mapped_itr = map(self.game_mapper, file_itr)
        return mapped_itr

        # todo reset_session?

    def get_indices(self, game):
        """
        Gamestring will be splitted by space. ==> List_of_list: [[e2],[e4]...]
        List_of_list will be sent to the tokenizer and converted to [[12,32],[12,23].....]
        This gives a list of list where each sublist is of different length (only for pgn)
        for uci the sublists are always (except the beginning annotation <|startoftext|>.....


        Then iterate the sublists and get the length (ignore dots).
        Length gives us the relevant indices of of the string. because the end of the sublist is equivalent to the
        ending of the move annotation.
        :param game:  UCI or PGN
        :return:
        """
        # todo chess_blindfolded
        input_ids = self.tokenizer(game["pgn"].split())["input_ids"] if self.notation == "pgn" else \
            self.tokenizer(game["uci"].split())["input_ids"]

        i = -1
        indices = []
        for token in input_ids:
            i += len(token)
            if token[-1] != 13:  # 13 resembles dot in pgn notation. ignores sublists/annotations like 1. 2. 3. usw.
                indices.append(i)
        return indices[2:]  # the first two are irrelevant since it contains the beginning <|startoftext|>....-0"]


def my_collate(batch):
    data = [item[0] for item in batch]
    v = {k: torch.cat([dic[k] for dic in data]) for k in data[0]}

    target = [item[1] for item in batch]
    target = [torch.flip(t, dims=[0]) for t in target]
    x = torch.LongTensor(pad_sequence(target, batch_first=True, padding_value=249))  # label will be 255
    target = torch.flip(x, dims=[1]) + 6  # add 6 so all labels are > 0

    n = target.shape[1]
    # Todo: check if this adds the correct amount of zeros to the ids
    #       check if tensor looks correct
    #       check if uci also correct
    x = [[0] * (n - len(item[-1])) + item[-1] for item in batch]
    ids = torch.LongTensor(x)  # pad ids with 0's

    return [v, target, ids]


if __name__ == "__main__":
    chess_models = [PgnGPT]
    test = False
    for model in chess_models:  # ["pretrained_gpt","pgn_gpt", "uci_gpt", "special_gpt"]
        # Creating the iterable dataset object
        # x = model.tokenizer('<|startoftext|>[Result "1-0"] 1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5'
        #                     ' 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 11. c4 c6 12. cxb5 axb5 13. Nc3 Bb7 14. Bg5 b4 '
        #                     '15. Nb1 h6 16. Bh4 c5 17. dxe5 Nxe4 18. Bxe7 Qxe7 19. exd6 Qf6 20. Nbd2 Nxd6 21. Nc4 Nxc4 '
        #                     '22. Bxc4 Nb6 23. Ne5 Rae8 24. Bxf7+ Rxf7 25. Nxf7 Rxe1+ 26. Qxe1 Kxf7 27. Qe3 Qg5 28. Qxg5'
        #                     ' hxg5 29. b3 Ke6 30. a3 Kd6 31. axb4 cxb4 32. Ra5 Nd5 33. f3 Bc8 34. Kf2 Bf5 35. Ra7 g6 '
        #                     '36. Ra6+ Kc5 37. Ke1 Nf4 38. g3 Nxh3 39. Kd2 Kb5 40. Rd6 Kc5 41. Ra6 Nf2 42. g4 Bd3 43. Re6 ',
        #                     return_tensors="pt", padding=True)
        # print(x["input_ids"])
        # input_ids = x["input_ids"]
        # y = model.tokenizer.convert_ids_to_tokens(x["input_ids"].squeeze())
        # print(y)
        # "O-O 9. h3 Nb8 10. d4 Nbd7 11. c4 c6 12. cxb5 axb5 13. Nc3 Bb7 14. Bg5 b4 15."
        # " Nb1 h6 16. Bh4 c5 17. dxe5 Nxe4 18. Bxe7 Qxe7 19. exd6 Qf6 20. Nbd2 Nxd6 21."
        # " Nc4 Nxc4 22. Bxc4 Nb6 23. Ne5 Rae8 24. Bxf7+ Rxf7 25. Nxf7 Rxe1+ 26. Qxe1 Kxf7"
        # " 27. Qe3 Qg5 28. Qxg5 hxg5 29. b3 Ke6 30. a3 Kd6 31. axb4 cxb4 32. Ra5 Nd5 33. "
        # "f3 Bc8 34. Kf2 Bf5 35. Ra7 g6 36. Ra6+ Kc5 37. Ke1 Nf4 38. g3 Nxh3 39. Kd2 Kb5 "
        # "40. Rd6 Kc5 41. Ra6 Nf2 42. g4 Bd3 43. Re6 ".split())

        # create data-loaders
        train_dataset = ChessGamesDataset("data/train_probing.jl", model.notation, model.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=2, collate_fn=my_collate, num_workers=4)
        val_dataset = ChessGamesDataset("data/val_probing.jl", model.notation, model.tokenizer)
        val_loader = DataLoader(val_dataset, batch_size=2, collate_fn=my_collate, num_workers=4)

        # create probing_classifier
        probing_classifier = ProbingChess(model)
        # create logger checkpoints
        tb_logger = pl_loggers.TensorBoardLogger(model.name + 'logs/')
        checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                              dirpath=model.name + 'pc_ckpt/',
                                              filename='sample-mnist-{epoch:02d}-{val_loss:.2f}')
        trainer = pl.Trainer(logger=tb_logger, gpus=0, limit_train_batches=10,
                             limit_val_batches=5,
                             val_check_interval=10,)

        trainer.fit(probing_classifier, train_loader, val_loader)

        # todo save and load model save_hyper-parameters
        if test:
            test_dataset = ChessGamesDataset("data/test_probing.jl", model.notation, model.tokenizer)
            test_loader = DataLoader(test_dataset, batch_size=8, collate_fn=my_collate, num_workers=16)
            trainer.test(test_dataloaders=test_loader)
