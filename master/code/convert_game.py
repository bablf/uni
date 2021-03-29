import chess.pgn as pgn
import pickle
import chess
import numpy as np
import random
import math

def preprocess_game_string(game, model_name, tokenizer):
    board = game.board()
    uci_move_list = []
    split_game = random.randint(0, len(list(game.mainline())) - 1)
    for i, move in enumerate(game.mainline_moves()):  # convert each game to uci with python chess
        if i == split_game:  # stop at the random point in pgn notation
            break
        uci_move_list.append(board.uci(move))
        board.push(move)
    split_game += math.ceil(split_game / 2)
    pgn_move_list = ' '.join(str(game.mainline()).split()[:split_game]) # cut pgn notation
    board = convert_board(board)  # convert board to format I need

    if model_name == "UciGPT":
        return '<|startoftext|>[Result "' + game.headers["Result"] + '"] ' + ' '.join(uci_move_list), board
    elif model_name == "PgnGPT":
        return '<|startoftext|>[Result "' + game.headers["Result"] + '"] ' + pgn_move_list, board
    elif model_name == "SpecialGPT":
        pass    # todo convert strings to human readable text with feldman_gpt2
                #   probably strip beginnnig tags
    else:  # == PretrainedGPT
        pass  # todo check what preprocessing is nessesary for this model



def convert_board(board):
    """
    Taken from python-chess/issues/404
    Converts the board to squared_list (see wikipedia)
    """
    l = [0] * 64
    for sq in chess.scan_reversed(board.occupied_co[chess.WHITE]):
        l[sq] = board.piece_type_at(sq)
    for sq in chess.scan_reversed(board.occupied_co[chess.BLACK]):
        l[sq] = -board.piece_type_at(sq)
    # print(np.array(l).reshape(8, 8))
    return np.array(l).reshape(64)


