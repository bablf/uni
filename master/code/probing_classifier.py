import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torchmetrics.functional import precision_recall, accuracy, f1
from pytorch_lightning.metrics.functional import confusion_matrix

from torch.optim import lr_scheduler


import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


class ProbingChess(pl.LightningModule):
    def __init__(self, chess_model):
        super().__init__()
        self.save_hyperparameters()
        self.chess_model = chess_model.model
        self.notation = chess_model.notation
        # Chess model does not need to keep track of gradients
        for param in self.chess_model.parameters():
            param.requires_grad = False

        self.dim_hidden_states = chess_model.config.n_embd
        self.batch_size, self.seq_length = 0, 0
        self.lr = 1e-6
        self.optimizer, self.scheduler = None, None

        # todo ALTERNATIVE SCHEDULAR smaller Train dataset for second schedular?
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

        # {2: "w_rook", 3: "w_bishop", 4: "w_knight", 1: "w_queen", 0: "w_king", 5: "w_pawn",
        #  10: "b_rook", 9: "b_bishop", 8: "b_knight", 11: "b_queen", 12: "b_king", 7: "b_pawn",
        #  6: "empty"}
        self.weights = torch.Tensor(
            [64 / 1, 64 / 1, 64 / 2, 64 / 2, 64 / 2, 64 / 8, 64 / 32, 64 / 8, 64 / 2, 64 / 2, 64 / 2, 64 / 1, 64 / 1]) \
            .cuda()  # give each peace a weight for CrossEntropyLoss Calculation.

        # layers of the model
        # 1024 -> 3000
        self.pre_linear_up_projection = nn.Linear(self.dim_hidden_states, 1024 * 10)  # todo activation relu & (dropout)
        # todo additional down/up procection
        self.linear_layers = nn.ModuleList([nn.Linear(1024 * 10, 13) for _ in range(64)])
        #self.lstm_layers = nn.ModuleList([nn.LSTM(input_size=1024*10, 1, 13) for _ in range(64)])

        self.relu = nn.ReLU()
        # self.gelu = nn.GELU()
        self.softmax = nn.LogSoftmax(dim=-1)
        # self.focal_losses = [FocalLoss(ignore_index=13) for _ in range(64)]
        self.losses = [nn.CrossEntropyLoss(weight=self.weights, ignore_index=13) for _ in range(64)]

    def forward(self, tokenized_chess_game, board, ids):
        # one forward pass
        self.chess_model.eval()  # Evaluation mode.
        with torch.no_grad():  # disable gradient calculation to save memory and requires_grad is False.
            gpt2_output = self.chess_model(input_ids=tokenized_chess_game["input_ids"],
                                           attention_mask=tokenized_chess_game["attention_mask"])
        # Extract dimensions from last_hidden_state that correspond to ending of move token.
        out = self.extract_relevant_dimensions(gpt2_output.last_hidden_state, ids)  # batch, seq_len 50-91 , 1024
        out = self.pre_linear_up_projection(out)
        out = torch.stack([layer(self.relu(out)) for layer in self.linear_layers])
        # out = torch.stack([layer(self.relu(out)) for layer in self.lstm_layers])
        # out = torch.stack([layer(self.gelu(out)) for layer in self.linear_layers])
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
        #final_loss.backward(retain_graph=True)

        # Logging to TensorBoard by default
        acc, precision, recall, f1_score = self.calc_metrics(prediction, board)

        self.log_dict({'train_loss': final_loss, 'train_acc': acc, 'train_recall': recall.item(),
                       'train_precision': precision.item(),
                       'train_f1_score': f1_score.item()}, logger=True, prog_bar=True, on_step=True, on_epoch=True)
        return final_loss

    def validation_step(self, batch, batch_idx):
        x, board, ids = batch
        self.batch_size, self.seq_length = board.shape[:2]
        output = self(x, board, ids)
        loss = self.loss(board, output)

        # calc loss and metrics

        acc, precision, recall, f1_score = self.calc_metrics(output, board)
        cfig_metrics, cpos_metrics = self.calc_chess_metrics(output, board)
        # log
        self.log_dict({'val_loss': loss, 'val_acc': acc, 'val_recall': recall.item(), 'val_precision': precision.item(),
                       'val_f1_score': f1_score.item()}, logger=True, prog_bar=True, on_epoch=True)

        return cfig_metrics, cpos_metrics, output, board
        # todo: log confusion matrix

    def validation_epoch_end(self, val_step_outputs):
        """
        cpos_metrics[0] (macro_average_acc, precision, recall, f1_score)
        :param val_step_outputs:
        :return:
        """
        self.logger.log_hyperparams(self.hparams)
        pass
        # # Todo how good is metric in beginning and end of game
        # num_batches = len(val_step_outputs)
        #
        # cfig = torch.stack([batch[0] for batch in val_step_outputs])
        # cpos = torch.stack([batch[1] for batch in val_step_outputs])
        # predictions = torch.stack([batch[2] for batch in val_step_outputs])
        # boards = torch.stack([batch[3] for batch in val_step_outputs])
        #
        # # # todo check if correct:
        # probs = self.softmax(predictions)
        # predictions = torch.argmax(probs, dim=-1).flatten()
        # boards = boards.flatten()
        # x = torch.where(boards != 13)  # filter unwanted indices.
        # conf_matrix = confusion_matrix(predictions[x], boards[x], num_classes=13)
        #
        # cfig = torch.sum(cfig, dim=0) / num_batches
        # cpos = torch.sum(cpos, dim=0) / num_batches
        #
        # acc_fig, pre_fig, rec_fig, f1_fig = self.create_images(cpos)

        # [10,  8,  9, 11, 12,  9,  8, 10,
        # 7,  7,  7,  7,  7,  7,  7,  7,
        # 6,  6, 6,  6,  6,  6,  6,  6,
        # 6,  6,  6,  6,  6,  6,  6,  6,
        # 6,  6,  6,  6, 6,  6,  6,  6,
        # 6,  6,  6,  6,  6,  6,  6,  6,
        # 5,  5,  5,  5,  5,  5,  5,  5,
        # 2,  4,  3,  1,  0,  3,  4,  2]
        num_2_piece = {2: "w_rook", 3: "w_bishop", 4: "w_knight", 1: "w_queen", 0: "w_king", 5: "w_pawn",
                       10: "b_rook", 9: "b_bishop", 8: "b_knight", 11: "b_queen", 12: "b_king", 7: "b_pawn",
                       6: "empty"}
        # # todo confusion matrix loggen
        # df_cm = pd.DataFrame(conf_matrix.detach().cpu().numpy(), index=range(), columns=range(10))
        # plt.figure(figsize=(10, 7))
        # fig_ = sn.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
        # plt.close(fig_)
        #
        # self.logger.experiment.add_figure("Chessboard_accuracy", acc_fig, self.current_epoch)
        # self.logger.experiment.add_figure("Chessboard_precision", pre_fig, self.current_epoch)
        # self.logger.experiment.add_figure("Chessboard_recall", rec_fig, self.current_epoch)
        # self.logger.experiment.add_figure("Chessboard_f1_score", f1_fig, self.current_epoch)
        # self.logger.experiment.add_figure("Chessboard_confusion_matrix", conf_fig, self.current_epoch)
        #
        # # metrics = dict(str,float)
        # # for acc, prec, recall, f1 etc
        # make X x 64 array out of cpos metrics
        # calc sum cpos_metric[:,i] / len(cpos_metric)

        # self.logger.log_metrics(dict())
        # self.log_dict({'val_epoch_cpos_acc': cpos_acc, 'val_epoch_cpos_prec': cpos_prec,
        # 'val_epoch_cpos_rec': cpos_rec,
        #             'val_epoch_cpos_f1': cpos_f1, "val_epoch_cfig_acc": cfig_acc,"val_epoch_cfig_prec": cfig_prec,
        #             "val_epoch_cfig_rec": cfig_rec, "val_epoch_cfig_f1": cfig_f1})

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
            # loss_values.append(self.focal_losses[i](predicted_board, true_board))

        return sum(loss_values)

    def calc_metrics(self, output, board):
        # get prediction
        probs = self.softmax(output)
        prediction = torch.argmax(probs, dim=-1).flatten()
        board = board.flatten()
        # calculate acc, precision & recall, f1

        macro_average_acc = accuracy(prediction, board, average='macro', num_classes=14, ignore_index=13)
        precision, recall = precision_recall(prediction, board, average="macro", num_classes=14, ignore_index=13)
        f1_score = f1(prediction, board, average="macro", num_classes=14, ignore_index=13)
        return macro_average_acc, precision, recall, f1_score

    def calc_chess_metrics(self, output, board):
        """
        chess_fig_metrics = Metrics (acc, precision, recall, f1) for each Chess figure
        chess_pos_metrics = Metrics (acc, precision, recall, f1) for each position (64) on the baord
        """
        # calc chess pos metrics
        chess_pos_metrics = torch.zeros(64 * 4).reshape(64, 4)
        for i in range(output.shape[2]):  # iterate over positions and calc metrics for each position
            chess_pos_metrics[i] = torch.stack(list(self.calc_metrics(output[:, :, i, :], board[:, :, i])))

        # get prediction
        probs = self.softmax(output)
        prediction = torch.argmax(probs, dim=-1).flatten()
        board = board.flatten()

        # calc chess figure metrics
        acc = accuracy(prediction, board, average="none", num_classes=14, ignore_index=13)  # shape(13,)
        precision, recall = precision_recall(prediction, board, average="none", num_classes=14,
                                             ignore_index=13)  # shape(
        f1_score = f1(prediction, board, average="none", num_classes=14, ignore_index=13)
        chess_fig_metrics = torch.stack([acc[:-1], precision[:-1], recall[:-1], f1_score[:-1]])

        return chess_fig_metrics, chess_pos_metrics

    def create_images(self, cpos):
        m = ["Accuracy", "Precision", "Recall", "F1 Score"]
        images = []
        for i, metric in enumerate(cpos.T):
            df = pd.DataFrame(metric.detach().cpu().numpy().reshape(8, 8),
                              index=[1, 2, 3, 4, 5, 6, 7, 8],
                              columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
            plt.figure()
            plt.title(m[i] + " of Chess Board Positions")
            plt.tick_params(labelbottom=False, bottom=False, top=False, labeltop=True)
            sn.set(font_scale=0.9)
            fig_ = sn.heatmap(df, annot=True, cmap="RdYlGn", linewidths=0.5).get_figure()
            # plt.show()
            images.append(fig_)
        return images

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = lr_scheduler.OneCycleLR(self.optimizer, max_lr=1e-6, epochs=self.trainer.min_epochs,
                                                 steps_per_epoch=self.trainer.limit_train_batches, pct_start=0.3)
        return [self.optimizer], [self.scheduler]

    def extract_relevant_dimensions(self, last_hidden_state, ids):
        """
        find the relevant indices from the tokenized game-string and return them.
        Search in second dimension for splitting of Moves.
        Relevant indices are created in Dataset and point at the end of each move.

        :last_hidden_state: shape = (batch, len(tokenized_chess_game), hidden) [2, 377, 1024]
        :return: shape = (batch, seq_length, hidden)
        """
        # extract indices from last_hidden_state # todo check if done right
        x = np.arange(last_hidden_state.shape[0])[:, None]  # [[0],[1],...]
        last_hidden_state = last_hidden_state[x, ids]
        return last_hidden_state

        # if self.notation == "pgn":
        #     n = board.shape[1]
        #     # extract indices from last_hidden_state
        #     last_hidden_state = last_hidden_state[np.arange(last_hidden_state.shape[0])[:, None], ids][:, -n:]
        #     return last_hidden_state
        #
        #     # for i, ids_ in enumerate(ids):
        #     #     last_hidden_state[i, ] = last_hidden_state[i, ids_, :]
        #
        #     # #thirteen = torch.where(last_hidden_state["input_ids"] == 13) - 2  # second move from movepair
        #     # input_ids = tokenized_chess_game["input_ids"]
        #     # y = []
        #     # for r in input_ids:
        #     #     y.append(self.tokenizer.convert_ids_to_tokens(r))
        #     #
        #     #
        #     # dot = torch.vstack(torch.where(input_ids == 13)).T
        #     # end_of_second_moves_indices = dot[:, 1] - 2
        #     #
        #     # z = [(y[i][j], input_ids[i, j].cpu().item()) for i, j in dot]
        #     # items = set(z)
        #     # print(items)
        #     #asdf = [pair for i, pair in enumerate(zip(y[0], input_ids[0].tolist()))] # for all batches
        #     #for p in asdf:
        #         #if p[0] in [']', '4', '7', '#', 'B', 'N', 'O', '3', '+', '1', 'R', '8', 'Q', '6', '2', '5'] or p[1] == ".":
        #         #16 - 23 46 10 49 45, 33, 48, 2, 12 ==> 16
        #
        #     # 16
        #     #all_indices = end_of_first_moves_indices + end_of_second_moves_indices
        #     #return last_hidden_state[all_indices]
        #         #   or find finde nachste Zahl nach punkt and try +1
        #             # check if numbers in between dots follow some pattern
        # else:
        #     # todo convert this to easier upper method
        #     n = board.shape[1]
        #     # extract indices from last_hidden_state # todo check if done right
        #     last_hidden_state = last_hidden_state[np.arange(last_hidden_state.shape[0])[:, None], ids][:, -n:]
        #     return last_hidden_state
        #
        #
        #
        #     n = board.shape[1]
        #
        #
        #
        #     batch_size = last_hidden_state.shape[0]
        #     length = last_hidden_state.shape[1]
        #     hid_dim = last_hidden_state.shape[2]
        #
        #     # input ids are > 200 if they indicate the beginning of a move
        #     x = torch.where(tokenized_chess_game["input_ids"] > 200)
        #     x = torch.stack(x, dim=1)  # stack the indices so we get [[0,0],[0,1] ... [15,249]]
        #     x[:, 1] = x[:, 1] - 1  # get the index before the number > 200 because we want the ending of the move
        #     y = x[:-1] != x[1:]  # this will give us a list of bool True False Pairs that shows if the batch value changed.
        #     # [False, True],[False, True],[False, True],[True, True],  # second value always true because index changes
        #     ind_of_batch = torch.where(y[:, 0])[0]  # find indices where first dim == True
        #     res = np.array_split(x.cpu(), list(ind_of_batch.cpu() + 1))  # split arrays into batches of relevant indices
        #
        #     final_ind = []
        #     for t in res:   # Todo ist vl unnötig
        #         final_ind.append(t[-n:])  # get the last n states that are relevant (because of board state)
        #     final_ind = torch.cat(final_ind)
        #     #  extract relevant dimensions and reshape.
        #     res = last_hidden_state[final_ind.T[0], final_ind.T[1], :].reshape(batch_size, n, hid_dim)
        #
        #     return res

# def test_step(self,  batch, batch_idx):
#     x, board, ids = batch
#     self.batch_size, self.seq_length = board.shape[:2]
#     output = self(x, board, ids)
#     # calc loss
#     loss = self.loss(board, output)
#     acc, precision, recall, f1_score, confusion_matrix = self.calc_metrics(output, board, confusion=True)
#     # log
#     self.log_dict({'test_loss': loss, 'test_acc': acc, 'test_recall': recall.item(),
#                    'test_precision': precision.item(), 'test_f1_score': f1_score.item()}, prog_bar=True, on_epoch=True)
#     # todo: log confusion matrix
#     # todo return accuracy per move
#
# def test_step_end(self, *args, **kwargs):
#
#     pass
#
# def test_epoch_end(self, outputs):
#     # todo alle return values von test step weiterverarbeiten
#     # todo berechne accuracy per move
#
#     pass
