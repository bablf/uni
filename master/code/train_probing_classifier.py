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
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch
from torch import from_numpy
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset, DataLoader

from models import PgnGPT, UciGPT
from probing_classifier import ProbingChess

seed_everything(40)


class ChessGamesDataset(IterableDataset):
    def __init__(self, filename, notation, tokenizer):
        self.filename = filename
        self.notation = notation
        self.tokenizer = tokenizer
        # get longest tokenized sequence so padding is possible
        if model.notation == "uci":
            self.tokenizer.model_max_length = model.tokenizer(open("data/max_length_uci.txt").readline(),
                                                              return_length=True)['length'] + 1
        else:  # == pgn
            self.tokenizer.model_max_length = model.tokenizer(open("data/max_length_pgn.txt").readline(),
                                                              return_length=True)['length'] + 1

    def game_mapper(self, file_iter):
        game = json.loads(file_iter)
        board = np.array(game["board"])
        tokenized_game = self.tokenizer(game["pgn"] if self.notation == "pgn" else game["uci"], return_tensors='pt',
                                        padding='max_length')
        indices = self.get_indices(game, tokenized_game, board.shape[0])
        # todo chess_blindfolded
        # if self.notation == "pgn":
        #     self.tokenizer(game["pgn"])
        # elif self.notation == "uci":
        #     self.tokenizer(game["uci"])
        # elif self.notation == "uci-blindfolded":
        #     self.tokenizer(" ".join(game["uci"].split()[2:]))

        return tokenized_game, from_numpy(board), indices

    def __iter__(self):
        # Create an iterator
        file_itr = open(self.filename, encoding='cp1252')
        # Map each element using the line_mapper
        mapped_itr = map(self.game_mapper, file_itr)
        return mapped_itr

        # todo reset_session?

    def get_indices(self, game, tokenized_game, n):
        """
        Gamestring will be splitted by space. ==> List_of_list: [[e2],[e4]...]
        List_of_list will be sent to the tokenizer and converted to [[12,32],[12,23].....]
        This gives a list of list where each sublist is of different length (only for pgn)
        for uci the sublists are always (except the beginning annotation <|startoftext|>.....


        Then iterate the sublists and get the length (ignore dots).
        Length gives us the relevant indices of of the string. because the end of the sublist is equivalent to the
        ending of the move annotation.
        :param n:
        :param tokenized_game:
        :param game:  UCI or PGN
        :return:
        """
        # todo chess_blindfolded
        # input_ids = self.tokenizer(game["pgn"].split())["input_ids"] if self.notation == "pgn" else \
        #     self.tokenizer(game["uci"].split())["input_ids"]
        token_list = self.tokenizer.convert_ids_to_tokens(tokenized_game["input_ids"][0])
        indices = []
        for i, _ in enumerate(tokenized_game["input_ids"][0]):
            if (token_list[i][0] == "Ġ" and token_list[i - 1] != '.') \
                    or token_list[i] == '<|endoftext|>':  # 13 resembles dot in pgn notation. ignores
                # sublists/annotations like 1. 2. 3. etc.
                if token_list[i - 1] != '"]':
                    indices.append(i - 1)

        indices.append(len(token_list) - 1)
        # indices.pop(1)  # remove second item doesn't give us any information for further usage.
        # z = sum(tokenized_game["input_ids"][0] == 50256)
        # indices = torch.LongTensor(indices) + z - 1  # add numb of padding ids to index numbers
        return indices  # the second is irrelevant since it contains the second part of the beginning string
        # we only need one for the beginning board state.


def my_collate(batch):
    data = [item[0] for item in batch]
    v = {k: torch.cat([dic[k] for dic in data]) for k in data[0]}

    target = [item[1] for item in batch]
    target = [torch.flip(t, dims=[0]) for t in target]
    x = torch.LongTensor(pad_sequence(target, batch_first=True, padding_value=7))  # label will be 13
    target = torch.flip(x, dims=[1]) + 6  # add 6 so all labels are > 0

    n = target.shape[1]
    ids = torch.vstack([torch.LongTensor(item[-1][-n:]) for item in batch])
    return [v, target, ids]
    # for item in batch:
    #     ids = torch.LongTensor(item[-1])
    #     length = len(item[-1])
    #     zeros = np.array([0] * (n - length))
    #    z = sum(item[0]["input_ids"][0] == 50256)

    #     ids += z
    #     ids = np.concatenate(zeros, ids)

    # x = [torch.cat([torch.LongTensor([0] * (n - len(item[-1]))),  # pad ids with 0's
    # torch.LongTensor(item[-1]) + sum(item[0]["input_ids"][0] == 50256) - 1])
    # add numb of padding ids to indix numbers
    #      for item in batch]
    # ids = torch.vstack(x)

    # x = [torch.cat([torch.LongTensor([0] * (n - len(item[-1]))),  # pad ids with 0's
    #                 torch.LongTensor(item[-1])])
    #      for item in batch]
    # ids = torch.vstack(x)


if __name__ == "__main__":
    chess_models = [UciGPT]
    test = False
    for model in chess_models:  # ["pretrained_gpt","pgn_gpt", "uci_gpt", "special_gpt"]
        # create data-loaders
        train_dataset = ChessGamesDataset("data/train_probing.jl", model.notation, model.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=my_collate, num_workers=16)
        val_dataset = ChessGamesDataset("data/val_probing.jl", model.notation, model.tokenizer)
        val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=my_collate, num_workers=16)
        # create probing_classifier
        probing_classifier = ProbingChess(model)

        # create logger
        tb_logger = pl_loggers.TensorBoardLogger(model.name + 'logs/')
        lr_monitor = LearningRateMonitor(logging_interval=None, log_momentum=True)
        checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                              dirpath=model.name + 'pc_ckpt/',
                                              filename='sample-{epoch:02d}-{val_loss:.2f}')
        # trainer = pl.Trainer(logger=tb_logger, gpus=1, overfit_batches=100,
        # limit_val_batches=1, check_val_every_n_epoch=1,
        #                     limit_train_batches=1, log_every_n_steps=10)
        # trainer = pl.Trainer(logger=tb_logger, gpus=1,
        #                      progress_bar_refresh_rate=100,
        #                      min_epochs=5,
        #                      log_every_n_steps=100,
        #                      val_check_interval=5000,
        #                      limit_val_batches=100,
        #                      )  # resume_from_checkpoint)
        trainer = pl.Trainer(logger=tb_logger, gpus=1,
                             progress_bar_refresh_rate=100,
                             min_epochs=5,
                             log_every_n_steps=100,
                             val_check_interval=200,
                             limit_train_batches=2000,
                             limit_val_batches=100,
                             callbacks=[lr_monitor],
                             auto_lr_find=True)

        trainer.tune(model=probing_classifier, train_dataloader=train_loader, val_dataloaders=val_loader,
                     lr_find_kwargs={"min_lr": 1e-15, "max_lr": 1e-3})
        trainer.fit(probing_classifier, train_loader, val_loader)

        # todo save and load model save_hyper-parameters
        if test:
            test_dataset = ChessGamesDataset("data/test_probing.jl", model.notation, model.tokenizer)
            test_loader = DataLoader(test_dataset, batch_size=8, collate_fn=my_collate, num_workers=16)
            trainer.test(test_dataloaders=test_loader)

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
