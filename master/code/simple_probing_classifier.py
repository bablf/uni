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
        self.num_classes = 1
        self.dim_hidden_states = chess_model.config.n_embd
        self.batch_size, self.seq_length = 0, 0
        self.lr = 1e-6
        self.optimizer, self.scheduler = None, None
        self.val_epoch = 0

        # {2: "w_rook", 3: "w_bishop", 4: "w_knight", 1: "w_queen", 0: "w_king", 5: "w_pawn",
        #  10: "b_rook", 9: "b_bishop", 8: "b_knight", 11: "b_queen", 12: "b_king", 7: "b_pawn",
        #  6: "empty"}
        # calculated weights from data_analysis.py
        # self.weights = torch.Tensor(
        #     [4.92307692, 6.24822549, 2.85111252, 3.5770647,  3.91028822, 0.77893641,
        #      0.12602016,
        #      0.77919055, 3.92469788, 3.59427825, 2.84273498, 6.23224483, 4.92307692]).cuda()

        # layers of the model
        self.pre_linear_up_projection = nn.Linear(self.dim_hidden_states, self.dim_hidden_states * 10)
        self.linear_layers = nn.ModuleList([nn.Linear(self.dim_hidden_states * 10, self.num_classes) for _ in range(64)])

        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=-1)
        self.losses = [nn.CrossEntropyLoss() for _ in range(64)]

    def forward(self, tokenized_chess_game, board, ids):
        # one forward pass
        self.chess_model.eval()  # Evaluation mode.
        with torch.no_grad():  # disable gradient calculation to save memory and requires_grad is False.
            if self.notation == "uci-blindfolded":
                gpt2_output = self.chess_model(input_ids=tokenized_chess_game["input_ids"])
                last_hidden_state = gpt2_output.hidden_states[-1]
            else:
                gpt2_output = self.chess_model(input_ids=tokenized_chess_game["input_ids"],
                                               attention_mask=tokenized_chess_game["input_ids"])
                last_hidden_state = gpt2_output.last_hidden_state

        # Extract dimensions from last_hidden_state that correspond to ending of move token.

        out = self.extract_relevant_dimensions(last_hidden_state, ids)  # batch, seq_len 50-91 , 10240
        # out = torch.rand(self.seq_length*self.dim_hidden_states).reshape(1, self.seq_length, self.dim_hidden_states).cuda()
        # out, _ = self.lstm(out)
        out = self.pre_linear_up_projection(out)
        out = torch.stack([layer(self.relu(out)) for layer in self.linear_layers])
        out = out.permute(1, 2, 0, 3)
        return out

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, board, ids = batch
        self.batch_size, self.seq_length = board.shape[:2]
        prediction = self(x, board, ids)

        # calculate loss
        final_loss = self.loss(board, prediction)

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
        cpos_metrics, cmoves_metrics = self.calc_chess_metrics(output, board)

        # log
        self.log_dict({'val_loss': loss, 'val_acc': acc, 'val_recall': recall.item(), 'val_precision': precision.item(),
                       'val_f1_score': f1_score.item()}, logger=True, prog_bar=True)

        # return output, board
        return cpos_metrics, output, board, cmoves_metrics

    def validation_epoch_end(self, val_step_outputs):
        """
        cpos_metrics[0] (macro_average_acc, precision, recall, f1_score)
        :param val_step_outputs:
        :return:
        """
        num_batches = len(val_step_outputs)

        cpos = torch.stack([batch[0] for batch in val_step_outputs])
        predictions = torch.cat([batch[1].flatten() for batch in val_step_outputs])
        boards = torch.cat([batch[2].flatten() for batch in val_step_outputs])
        max_l = max([len(batch[3]) for batch in val_step_outputs])
        cmoves = torch.stack([torch.stack(batch[3]) for batch in val_step_outputs if len(batch[3]) == max_l])

        probs = self.citerion(predictions)
        boards = torch.flatten(boards)
        conf_matrix = confusion_matrix(probs, boards, num_classes=2).detach().cpu()

        # calc mean
        cpos = torch.sum(cpos, dim=0) / num_batches
        cmoves = torch.sum(cmoves, dim=0) / len(cmoves)

        acc_fig, pre_fig, rec_fig, f1_fig = self.create_images(cpos, heatmap=True)
        acc_moves, pre_moves, rec_moves, f1_moves = self.create_images(cmoves, max_l=max_l)

        group_counts = ["{0:0.0f}".format(value) for value in conf_matrix.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in conf_matrix.flatten() / torch.sum(conf_matrix)]
        labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)

        _ = plt.figure()
        _ = plt.title("Binary Confusion Matrix")
        _ = sn.set(font_scale=0.9)
        fig_ = sn.heatmap(conf_matrix, annot=labels, cmap='Blues', linewidths=0.7, fmt='').get_figure()
        _ = plt.close(fig_)

        self.logger.experiment.add_figure("Confusion matrix", fig_, self.val_epoch)
        self.logger.experiment.add_figure("Chessboard_accuracy", acc_fig, self.val_epoch)
        self.logger.experiment.add_figure("Chessboard_precision", pre_fig, self.val_epoch)
        self.logger.experiment.add_figure("Chessboard_recall", rec_fig, self.val_epoch)
        self.logger.experiment.add_figure("Chessboard_f1_score", f1_fig, self.val_epoch)

        self.logger.experiment.add_figure("Move_accuracy", acc_moves, self.val_epoch)
        self.logger.experiment.add_figure("Move_precision", pre_moves, self.val_epoch)
        self.logger.experiment.add_figure("Move_recall", rec_moves, self.val_epoch)
        self.logger.experiment.add_figure("Move_f1_score", f1_moves, self.val_epoch)
        self.val_epoch += 1

    def loss(self, board, prediction):
        loss_values = []
        for i in range(64):
            # extract relevant field in batch/sequence_length
            true_board = board[:, :, i]  # shape(batch_size, len)
            predicted_board = prediction[:, :, i, :]  # shape(batch_size, len, 13)
            # reshape boards
            predicted_board = predicted_board.reshape(self.batch_size * self.seq_length, self.num_classes)
            true_board = true_board.reshape(self.batch_size * self.seq_length)
            # calculate losses
            loss_values.append(self.losses[i](predicted_board, true_board))

        return sum(loss_values)

    def calc_metrics(self, output, board):
        # get prediction
        probs = self.softmax(output)
        prediction = torch.argmax(probs, dim=-1).flatten()
        # prediction = torch.zeros(self.batch_size*self.seq_length*64, dtype=torch.int64).flatten().cuda()
        board = board.flatten()

        # calculate acc, precision & recall, f1
        # prediction = torch.argmax(probs, dim=-1)[:, :, 0].flatten()
        # board = board[:, :, 0].flatten()

        # macro_average_acc = accuracy(prediction, board, average='micro', num_classes=self.num_classes)[0]
        # precision, recall = precision_recall(prediction, board, average="micro", num_classes=self.num_classes)
        # f1_score = f1(prediction, board, average="micro", num_classes=self.num_classes)[0]

        macro_average_acc = accuracy(prediction, board, average='macro', num_classes=self.num_classes)
        precision, recall = precision_recall(prediction, board, average="macro", num_classes=self.num_classes)
        f1_score = f1(prediction, board, average="macro", num_classes=self.num_classes)
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
        acc = accuracy(prediction, board, average="none", num_classes=self.num_classes)  # shape(self.num_classes,)
        precision, recall = precision_recall(prediction, board, average="none", num_classes=self.num_classes)  # shape(
        f1_score = f1(prediction, board, average="none", num_classes=self.num_classes)
        chess_fig_metrics = torch.stack([acc[:-1], precision[:-1], recall[:-1], f1_score[:-1]])

        return chess_fig_metrics, chess_pos_metrics

    def create_images(self, cpos, heatmap=False, max_l=0):
        m = ["Accuracy", "Precision", "Recall", "F1 Score"]
        images = []
        for i, metric in enumerate(cpos.T):

            if heatmap:
                df = pd.DataFrame(metric.detach().cpu().numpy().reshape(8, 8),
                                  index=[1, 2, 3, 4, 5, 6, 7, 8],
                                  columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
                                  )
                _ = plt.figure()
                _ = plt.title(m[i] + " of Chess Board Positions")
                _ = plt.tick_params(labelbottom=False, bottom=False, top=False, labeltop=True)
                _ = sn.set(font_scale=0.9)

                fig_ = sn.heatmap(df, annot=True, cmap="RdYlGn", linewidths=0.7, vmin=0.0, vmax=1.0).get_figure()
                _ = plt.close(fig_)
                images.append(fig_)

            else:  # lineplot
                df = pd.DataFrame(metric.detach().cpu().numpy(), index=range(max_l), columns=[m[i]])
                df.reset_index(inplace=True)
                fig, ax = plt.subplots()
                ax.set_ylim(0.0, 1.05)
                _ = plt.title(m[i] + " of " + "Chess Board at Move x")
                _ = sn.set(font_scale=0.9)
                fig_ = sn.lineplot(data=df, x="index", y=m[i]).get_figure()
                _ = plt.close(fig_)
                images.append(fig_)
        return images

    def configure_optimizers(self):
        max_lr = 1e-6
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=1,
                                        # threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
        # self.scheduler = lr_scheduler.OneCycleLR(self.optimizer, max_lr=max_lr, epochs=self.trainer.min_epochs,
        #                                          steps_per_epoch=self.trainer.limit_train_batches, pct_start=0.3)
        # self.add_to_hparams("schedular", self.scheduler.__class__.__name__)
        # self.add_to_hparams("schedular max_lr", max_lr)

        # return [self.optimizer], [self.scheduler]
        return self.optimizer

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

    def add_to_hparams(self, name, value):
        self.hyperparams[name] = value

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
