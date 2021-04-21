import random

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import trainer
from torchmetrics.functional import precision_recall, accuracy, f1



class ProbingChess(pl.LightningModule):
    def __init__(self, chess_model, dataset):
        super().__init__()
        self.chess_model = chess_model.model
        self.notation = chess_model.notation
        # Chess model does not need to keep track of gradients
        for param in self.chess_model.parameters():
            param.requires_grad = False

        self.dim_hidden_states = chess_model.config.n_embd
        self.seq_length = dataset.tokenizer.model_max_length
        self.tokenizer = dataset.tokenizer
        # layers of the model

        self.linear = nn.Linear(self.dim_hidden_states, 64*13)
        self.linear_layers = nn.ModuleList([nn.Linear(self.dim_hidden_states, 13) for _ in range(64)])
        self.softmax = nn.LogSoftmax(dim=-1)
        self.losses = [nn.CrossEntropyLoss(ignore_index=255) for _ in range(64)]

    def forward(self, tokenized_chess_game, board, ids):
        # one forward pass
        self.chess_model.eval()  # Evaluation mode.
        with torch.no_grad():  # disable gradient calculation to save memory and requires_grad is False.
            gpt2_output = self.chess_model(input_ids=tokenized_chess_game["input_ids"],
                                           attention_mask=tokenized_chess_game["attention_mask"])
        # Extract dimensions from last_hidden_state that correspond to ending of move token.
        relevant_dimensions = self.extract_relevant_dimensions(gpt2_output.last_hidden_state, tokenized_chess_game, board, ids)
        out = torch.stack([layer(relevant_dimensions) for layer in self.linear_layers])
        out = out.reshape(self.batch_size, self.seq_length, 64, 13)
        return out


    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, board, ids = batch
        self.batch_size, self.seq_length = board.shape[:2]
        prediction = self(x, board, ids)

        # calculate loss
        final_loss = self.loss(board, prediction)
        final_loss.backward(retain_graph=True)

        # Logging to TensorBoard by default
        self.log('train_loss', final_loss)
        return final_loss

    def validation_step(self, batch, batch_idx):
        x, board, ids = batch
        self.batch_size, self.seq_length = board.shape[:2]
        # implement your own
        output = self(x, board, ids)
        # calc loss
        loss = self.loss(board, output)
        # make predictions
        probs = self.softmax(output)
        prediction = torch.argmax(probs, dim=-1).flatten()
        board = board.flatten()

        # calculate acc
        acc = accuracy(prediction, board)
        # calc precision & recall
        precision, recall = precision_recall(prediction, board, average="micro", ignore_index=255)
        # calc f1
        f1_score = f1(prediction, board, average="micro", ignore_index=255)  # macro für imbalance an daten
        # log the outputs

        self.log_dict({'val_loss': loss, 'val_acc': acc, 'val_recall': recall.item(), 'val_precision': precision.item(),
                       'val_f1_score': f1_score.item()}, prog_bar=True,
                      )

    def loss(self, board, prediction):
        loss_values = []
        for i in range(64):
            # extract relevant field in batch/sequence_length
            true_board = board[:, :, i]  # shape(batch_size, len)
            predicted_board = prediction[:, :, i, :]  # shape(batch_size, len, 13)
            # reshape boards
            predicted_board = predicted_board.reshape(self.batch_size * self.seq_length, 13)
            true_board = true_board.reshape(self.batch_size * self.seq_length)
            # calculate losses
            loss_values.append(self.losses[i](predicted_board, true_board))

        return sum(loss_values)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def extract_relevant_dimensions(self, last_hidden_state, tokenized_chess_game, board, ids):
        """
        find the relevant indices from the tokenized gamestring and return them.
        Search in second dimension for splitting of Moves.
        :last_hidden_state: shape = (batch, seq_len(padded), hidden) [2, 377, 1024]
        :return: shape = (batch, board_padding, hidden)
        Todo:   might be more difficult for PGN
                problem mit anfang
        """
        if self.notation == "pgn":
            max()
            for i, ids_ in enumerate(ids):
                last_hidden_state[i, ] = last_hidden_state[i, ids_, :]

            # #thirteen = torch.where(last_hidden_state["input_ids"] == 13) - 2  # second move from movepair
            # input_ids = tokenized_chess_game["input_ids"]
            # y = []
            # for r in input_ids:
            #     y.append(self.tokenizer.convert_ids_to_tokens(r))
            #
            #
            # dot = torch.vstack(torch.where(input_ids == 13)).T
            # end_of_second_moves_indices = dot[:, 1] - 2
            #
            # z = [(y[i][j], input_ids[i, j].cpu().item()) for i, j in dot]
            # items = set(z)
            # print(items)
            #asdf = [pair for i, pair in enumerate(zip(y[0], input_ids[0].tolist()))] # for all batches
            #for p in asdf:
                #if p[0] in [']', '4', '7', '#', 'B', 'N', 'O', '3', '+', '1', 'R', '8', 'Q', '6', '2', '5'] or p[1] == ".":
                #16 - 23 46 10 49 45, 33, 48, 2, 12 ==> 16

            # 16
            end_of_first_moves_indices = None

            #all_indices = end_of_first_moves_indices + end_of_second_moves_indices
            #return last_hidden_state[all_indices]
            # todo equal 13 and get second space after 13
                #   or find finde nachste Zahl nach punkt and try +1
                    # check if numbers in between dots follow some pattern
        else:
            n = board.shape[1]
            batch_size = last_hidden_state.shape[0]
            length = last_hidden_state.shape[1]
            hid_dim = last_hidden_state.shape[2]

            # input ids are > 200 if they indicate the beginning of a move
            x = torch.where(tokenized_chess_game["input_ids"] > 200)
            x = torch.stack(x, dim=1)  # stack the indices so we get [[0,0],[0,1] ... [15,249]]
            x[:, 1] = x[:, 1] - 1  # get the index before the number > 200 because we want the ending of the move
            y = x[:-1] != x[1:]  # this will give us a list of bool True False Pairs that shows if the batch value changed.
            # [False, True],[False, True],[False, True],[True, True],  # second value always true because index changes
            ind_of_batch = torch.where(y[:, 0])[0]  # find indices where first dim == True
            res = np.array_split(x.cpu(), list(ind_of_batch.cpu() + 1))  # split arrays into batches of relevant indices

            final_ind = []
            for t in res:   # Todo ist vl unnötig
                final_ind.append(t[-n:])  # get the last n states that are relevant (because of board state)
            final_ind = torch.cat(final_ind)
            #  extract relevant dimensions and reshape.
            res = last_hidden_state[final_ind.T[0], final_ind.T[1], :].reshape(batch_size, n, hid_dim)

            return res




























