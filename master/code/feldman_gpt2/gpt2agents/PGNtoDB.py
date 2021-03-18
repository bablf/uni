import os
import random
import re
import shlex
import datetime
import glob
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
from typing import List, Dict, Tuple, TextIO, Pattern

import gpt2agents.utils.MySqlInterface as MSI

"""
Portable game notation from https://en.wikipedia.org/wiki/Portable_Game_Notation

For most moves the SAN consists of the letter abbreviation for the piece, an x if there is a capture, and the 
two-character algebraic name of the final square the piece moved to. The letter abbreviations are K (king), 
Q (queen), R (rook), B (bishop), and N (knight). The pawn is given an empty abbreviation in SAN movetext, but 
in other contexts the abbreviation P is used. The algebraic name of any square is as per usual algebraic chess 
notation; from white's perspective, the leftmost square closest to white is a1, the rightmost square closest to the 
white is h1, and the rightmost (from white's perspective) square closest to black side is h8.

In a few cases a more detailed representation is needed to resolve ambiguity; if so, the piece's file letter, 
numerical rank, or the exact square is inserted after the moving piece's name (in that order of preference). 
Thus, Nge2 specifies that the knight originally on the g-file moves to e2.

SAN kingside castling is indicated by the sequence O-O; queenside castling is indicated by the sequence O-O-O 
(note that these are capital Os, not zeroes, contrary to the FIDE standard for notation). Pawn promotions are 
notated by appending = to the destination square, followed by the piece the pawn is promoted to. For example: 
e8=Q. If the move is a checking move, + is also appended; if the move is a checkmating move, # is appended instead. 
For example: e8=Q#.
"""

print_state = 0
def dprint(s:str, threshold:int = 0):
    if print_state > 0:
        print(s)

class PIECES(Enum):
    NONE = ""
    WHITE_KING = "white king"  # K
    WHITE_QUEEN = "white queen"  # Q
    WHITE_BISHOP = "white bishop"  # B
    WHITE_KNIGHT = "white knight"  # N
    WHITE_ROOK = "white rook"  # R
    WHITE_PAWN = "white pawn"  # empty
    BLACK_KING = "black king"  # k
    BLACK_QUEEN = "black queen"  # q
    BLACK_BISHOP = "black bishop"  # b
    BLACK_KNIGHT = "black knight"  # n
    BLACK_ROOK = "black rook"  # r
    BLACK_PAWN = "black pawn"  # empty
    KING = "king"  # k
    QUEEN = "queen"  # q
    BISHOP = "bishop"  # b
    KNIGHT = "knight"  # n
    ROOK = "rook"  # r
    PAWN = "pawn"  # empty


class CASTLING(Enum):
    NO = "no"
    KINGSIDE = "kingside castles"
    QUEENSIDE = "queenside castles"


class RESULTS(Enum):
    CONTINUE = "continue"
    WIN = "wins"
    LOSE = "resigns"
    DRAW = "declares a draw"


class Chessboard():
    board: pd.DataFrame
    num_index: List
    char_index: List

    def __init__(self):
        self.reset()

    def reset(self):
        self.char_index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.num_index = [8, 7, 6, 5, 4, 3, 2, 1]
        self.board = df = pd.DataFrame(columns=self.char_index, index=self.num_index)
        for number in self.num_index:
            for letter in self.char_index:
                df.at[number, letter] = PIECES.NONE.value

        self.populate_board()
        # self.print_board()

    def populate_board(self):
        self.set_piece_at(('a', 1), PIECES.WHITE_ROOK)
        # self.board.at[1, 'a'] = PIECES.WHITE_ROOK.value
        self.board.at[1, 'h'] = PIECES.WHITE_ROOK.value
        self.board.at[1, 'b'] = PIECES.WHITE_KNIGHT.value
        self.board.at[1, 'g'] = PIECES.WHITE_KNIGHT.value
        self.board.at[1, 'c'] = PIECES.WHITE_BISHOP.value
        self.board.at[1, 'f'] = PIECES.WHITE_BISHOP.value
        self.board.at[1, 'd'] = PIECES.WHITE_QUEEN.value
        self.board.at[1, 'e'] = PIECES.WHITE_KING.value

        self.board.at[8, 'a'] = PIECES.BLACK_ROOK.value
        self.board.at[8, 'h'] = PIECES.BLACK_ROOK.value
        self.board.at[8, 'b'] = PIECES.BLACK_KNIGHT.value
        self.board.at[8, 'g'] = PIECES.BLACK_KNIGHT.value
        self.board.at[8, 'c'] = PIECES.BLACK_BISHOP.value
        self.board.at[8, 'f'] = PIECES.BLACK_BISHOP.value
        self.board.at[8, 'd'] = PIECES.BLACK_QUEEN.value
        self.board.at[8, 'e'] = PIECES.BLACK_KING.value

        for letter in self.char_index:
            self.board.at[2, letter] = PIECES.WHITE_PAWN.value
            self.board.at[7, letter] = PIECES.BLACK_PAWN.value

    def check_if_clear(self, loc:Tuple, candidate:Tuple, piece:PIECES) -> bool:
        if piece == PIECES.WHITE_PAWN or piece == PIECES.BLACK_PAWN:
            return True
        if piece == PIECES.WHITE_KNIGHT or piece == PIECES.BLACK_KNIGHT:
            return True
        if piece == PIECES.WHITE_KING or piece == PIECES.BLACK_KING:
            return True

        c_col_i = self.char_index.index(candidate[0])
        c_row_i = self.num_index.index(candidate[1])
        l_col_i = self.char_index.index(loc[0])
        l_row_i = self.num_index.index(loc[1])
        col_dist = c_col_i - l_col_i
        row_dist = c_row_i - l_row_i
        dist = max(abs(col_dist), abs(row_dist))
        col_vec = 0
        row_vec = 0
        if col_dist != 0:
            col_vec = int(col_dist/dist)
        if row_dist != 0:
            row_vec = int(row_dist/dist)

        col_i = l_col_i + col_vec
        row_i = l_row_i + row_vec
        for i in range(dist-1):
            num = self.num_index[row_i]
            char = self.char_index[col_i]
            pos = (char, num)
            p = self.get_piece_at(pos)
            if p != PIECES.NONE:
                dprint("Chessboard.check_if_clear() {} is blocked by {}".format(piece.value, p.value))
                return False
            col_i += col_vec
            row_i += row_vec

        return True

    def find_first_piece(self, loc: Tuple, offset_list: List, piece: PIECES, hint: str = '*') -> Tuple:


        for t in offset_list:

            try:
                ci = self.char_index.index(loc[0]) + t[0]
                ni = self.num_index.index(loc[1]) - t[1]
                if ci < 0 or ni < 0:
                    # dprint("{} offset {} -> ({}, {}) out of bounds".format(piece.value, t, ci, ni))
                    continue
                num = self.num_index[ni]
                char = self.char_index[ci]
                if hint == '*' or hint == char or int(hint) == num:
                    candidate = (char, num)
                    p = self.get_piece_at(candidate)
                    #dprint("{}: found [{}] at ({}, {})/{}".format(piece.value, p.value, char, num, t))
                    if p == piece:
                        if self.check_if_clear(loc, candidate, piece):
                            #dprint("\tMatch! Moving {} from ({}, {}) to {}".format(p.value, char, num, loc))
                            return candidate
            except IndexError:
                # dprint("IndexError: {} offset {} out of bounds".format(piece.value, t))
                pass
            except ValueError:
                # dprint("ValueError: {} offset = {}, loc = {}".format(piece.value, t, loc))
                pass
        dprint("Chessboard.find_first_piece() problem with {} moving to {}".format(piece.value, loc))
        # self.print_board()
        return (-1, -1)

    # use the letter/number traditional chess indexing (e.g. Qb4)
    def get_piece_at(self, loc: Tuple) -> PIECES:
        r = tuple(reversed(loc))
        piece = self.board.at[r]
        return PIECES(piece)

    # use the letter/number traditional chess indexing (e.g. Qb4)
    def set_piece_at(self, loc: Tuple, piece: PIECES):
        r = tuple(reversed(loc))
        self.board.at[r] = piece.value

    def print_board(self, override:bool=False):
        if override or print_state > 1:
            fig, ax = plt.subplots()

            # hide axes
            fig.patch.set_visible(False)
            ax.axis('off')
            ax.axis('tight')

            ax.table(cellText=self.board.values, colLabels=self.char_index, rowLabels=self.num_index, loc='center')

            fig.tight_layout()

            plt.show()


class Move:
    white_raw: str
    black_raw: str
    white_expanded: str
    black_expanded: str
    white_val_dict: Dict
    black_val_dict: Dict
    comment_list: List
    white_castles: CASTLING
    black_castles: CASTLING
    is_end: bool
    has_problem: bool
    board: Chessboard
    white_player: str
    black_player: str
    piece_regex: Pattern
    square_regex: Pattern
    num_regex: Pattern
    alpha_regex: Pattern
    hints_regex: Pattern
    comment_regex: Pattern
    no_comment_regex: Pattern

    # /[^{\}]+(?=})/g, from https://stackoverflow.com/questions/413071/regex-to-get-string-between-curly-braces
    def __init__(self, line: str, white: str, black: str, board: Chessboard, comment_list: List):
        self.reset()
        self.comment_list = comment_list
        self.board = board
        self.white_player = white
        self.black_player = black
        line = line.strip()
        dprint("Evaluating move [{}]".format(line))

        moves = line.split()
        # if white resigns or if it's a draw
        result = self.test_end(moves[0], True)
        if result != RESULTS.CONTINUE:
            self.white_expanded = "{} {}.".format(self.white_player, result.value)
            result = self.test_end(moves[0], False)
            self.black_expanded = "{} {}".format(self.black_player, result.value)
            self.is_end = True
            return
        # get the move
        self.white_raw = moves[0]
        self.evaluate_move(self.white_raw, True)

        # if black resigns
        result = self.test_end(moves[1], False)
        if result != RESULTS.CONTINUE:
            self.black_expanded = "{} {}.".format(self.black_player, result.value)
            result = self.test_end(moves[1], True)
            self.white_expanded = "{} {}.".format(self.white_player, result.value)
            self.is_end = True
            return
        self.black_raw = moves[1]
        self.evaluate_move(self.black_raw, False)

        # check to see if an end condition has crept through
        result = self.test_end(line, True)
        if result != RESULTS.CONTINUE:
            self.white_expanded = "{} {}.".format(self.white_player, result.value)
            result = self.test_end(line, False)
            self.black_expanded = "{} {}".format(self.black_player, result.value)
            self.is_end = True

        # dprint(self.to_string())
        self.board.print_board()

    def reset(self):
        self.is_end = False
        self.has_problem = False
        self.white_raw = "resign"
        self.black_raw = "resign"
        self.white_expanded = "unset"
        self.black_expanded = "unset"
        self.white_val_dict = {}
        self.black_val_dict = {}
        self.comment = ""
        self.board = None
        self.piece_regex = re.compile('[^A-Z]')
        self.square_regex = re.compile('[A-Z]')
        self.num_regex = re.compile('[^0-9]')
        self.alpha_regex = re.compile('[0-9]')
        self.hints_regex = re.compile('[KQNBRx+#=]')
        self.white_castles = CASTLING.NO
        self.black_castles = CASTLING.NO

    def test_end(self, s: str, is_white: bool) -> RESULTS:
        ends = ['1-0', '0-1', '1/2-1/2']
        if is_white:
            if ends[0] in s:
                return RESULTS.WIN
            elif ends[1] in s:
                return RESULTS.LOSE
            elif ends[2] in s:
                return RESULTS.DRAW
        else:  # black
            if ends[0] in s:
                return RESULTS.LOSE
            elif ends[1] in s:
                return RESULTS.WIN
            elif ends[2] in s:
                return RESULTS.DRAW
        return RESULTS.CONTINUE

    def create_bishop_offsets(self) -> List:
        # go from (1, 1), (-1, 1), (-1, -1), (1, -1) to (8, 8), (-8, 8), (-8, -8), (8, -8)
        to_return = []
        for i in range(1, 8):
            to_return.append((i, i))
            to_return.append((-i, i))
            to_return.append((-i, -i))
            to_return.append((i, -i))
        return to_return

    def create_rook_offsets(self) -> List:
        # go from (1, 0), (-1, 0), (0, 1), (0, -1) to (8, 0), (-8, 0), (0, 8), (0, -8)
        to_return = []
        for i in range(1, 8):
            to_return.append((i, 0))
            to_return.append((-i, 0))
            to_return.append((0, i))
            to_return.append((0, -i))
        return to_return

    # we can have more than one queen if a rook gets promoted
    def create_queen_offsets(self) -> List:
        # go from (1, 1), (-1, 1), (-1, -1), (1, -1),(1, 0), (-1, 0), (0, 1), (0, -1) to (8, 8), (-8, 8), (-8, -8), (8, -8), (8, 0), (-8, 0), (0, 8), (0, -8)
        to_return = []
        for i in range(1, 8):
            to_return.append((i, i))
            to_return.append((-i, i))
            to_return.append((-i, -i))
            to_return.append((i, -i))
            to_return.append((i, 0))
            to_return.append((-i, 0))
            to_return.append((0, i))
            to_return.append((0, -i))
        return to_return

    def find_closest_piece(self, piece: PIECES, destination: Tuple, is_white: bool, takes: bool = False,
                           hint: str = '*') -> Tuple:
        to_return = (-1, -1)
        if is_white:
            if piece == PIECES.PAWN:
                offset_list = [(0, -1), (0, -2)]
                if takes:
                    offset_list = [(0, -1), (0, -2), (-1, -1), (1, -1)]
                to_return = self.board.find_first_piece(destination, offset_list, PIECES.WHITE_PAWN, hint=hint)
            elif piece == PIECES.KNIGHT:
                offset_list = [(-1, 2), (1, 2), (-1, -2), (1, -2), (-2, -1), (2, -1), (-2, 1), (2, 1)]
                to_return = self.board.find_first_piece(destination, offset_list, PIECES.WHITE_KNIGHT, hint=hint)
            elif piece == PIECES.BISHOP:
                to_return = self.board.find_first_piece(destination, self.create_bishop_offsets(), PIECES.WHITE_BISHOP,
                                                        hint=hint)
            elif piece == PIECES.ROOK:
                to_return = self.board.find_first_piece(destination, self.create_rook_offsets(), PIECES.WHITE_ROOK,
                                                        hint=hint)
            elif piece == PIECES.QUEEN:
                to_return = self.board.find_first_piece(destination, self.create_queen_offsets(), PIECES.WHITE_QUEEN,
                                                        hint=hint)
            elif piece == PIECES.KING:
                offset_list = [(1, 1), (1, 0), (1, -1), (-1, 1), (-1, 0), (-1, -1), (0, 1), (0, -1)]
                to_return = self.board.find_first_piece(destination, offset_list, PIECES.WHITE_KING, hint=hint)
        if not is_white:
            if piece == PIECES.PAWN:
                offset_list = [(0, 1), (0, 2)]
                if takes:
                    offset_list = [(0, 1), (0, 2), (-1, 1), (1, 1)]
                to_return = self.board.find_first_piece(destination, offset_list, PIECES.BLACK_PAWN, hint=hint)
            elif piece == PIECES.KNIGHT:
                offset_list = [(-1, 2), (1, 2), (-1, -2), (1, -2), (-2, -1), (-2, 1), (2, 1), (2, -1)]
                to_return = self.board.find_first_piece(destination, offset_list, PIECES.BLACK_KNIGHT, hint=hint)
            elif piece == PIECES.BISHOP:
                to_return = self.board.find_first_piece(destination, self.create_bishop_offsets(), PIECES.BLACK_BISHOP,
                                                        hint=hint)
            elif piece == PIECES.ROOK:
                to_return = self.board.find_first_piece(destination, self.create_rook_offsets(), PIECES.BLACK_ROOK,
                                                        hint=hint)
            elif piece == PIECES.QUEEN:
                to_return = self.board.find_first_piece(destination, self.create_queen_offsets(), PIECES.BLACK_QUEEN,
                                                        hint=hint)
            elif piece == PIECES.KING:
                offset_list = [(1, 1), (1, 0), (1, -1), (-1, 1), (-1, 0), (-1, -1), (0, 1), (0, -1)]
                to_return = self.board.find_first_piece(destination, offset_list, PIECES.BLACK_KING, hint=hint)

        if to_return == (-1, -1):
            dprint("Move.find_closest_piece() Problem with {} moving to{} ".format(piece.value, destination))
        return to_return

    def evaluate_move(self, move_str: str, is_white: bool):
        # see what the piece is and where it is
        piece = self.piece_regex.sub('', move_str)
        dprint("piece string = '{}' (blank is pawn)".format(piece))
        source_loc = (-1, -1)
        p = PIECES.NONE
        self.black_castles = CASTLING.NO
        self.white_castles = CASTLING.NO
        if piece == 'OO':  # castling is a special case
            if is_white:
                self.white_castles = CASTLING.KINGSIDE
                self.board.set_piece_at(('e', 1), PIECES.NONE)
                self.board.set_piece_at(('h', 1), PIECES.NONE)
                self.board.set_piece_at(('f', 1), PIECES.WHITE_ROOK)
                self.board.set_piece_at(('g', 1), PIECES.WHITE_KING)
            else:
                self.black_castles = CASTLING.KINGSIDE
                self.board.set_piece_at(('e', 8), PIECES.NONE)
                self.board.set_piece_at(('h', 8), PIECES.NONE)
                self.board.set_piece_at(('f', 8), PIECES.BLACK_ROOK)
                self.board.set_piece_at(('g', 8), PIECES.BLACK_KING)

            self.set_expanded(source_loc, source_loc, p, is_white)
            return
        elif piece == 'OOO':  # castling is a special case
            if is_white:
                self.white_castles = CASTLING.QUEENSIDE
                self.board.set_piece_at(('e', 1), PIECES.NONE)
                self.board.set_piece_at(('a', 1), PIECES.NONE)
                self.board.set_piece_at(('d', 1), PIECES.WHITE_ROOK)
                self.board.set_piece_at(('c', 1), PIECES.WHITE_KING)
            else:
                self.black_castles = CASTLING.QUEENSIDE
                self.board.set_piece_at(('e', 8), PIECES.NONE)
                self.board.set_piece_at(('a', 8), PIECES.NONE)
                self.board.set_piece_at(('d', 8), PIECES.BLACK_ROOK)
                self.board.set_piece_at(('c', 8), PIECES.BLACK_KING)

            self.set_expanded(source_loc, source_loc, p, is_white)
            return

        # Now do the other cases
        hint = '*'
        takes = False
        cleaned = self.hints_regex.sub('', move_str)
        if len(cleaned) > 2:
            hint = cleaned[0]
            cleaned = cleaned[1:]
        piece = self.piece_regex.sub('', move_str)
        num = self.num_regex.sub('', cleaned)
        letter = self.alpha_regex.sub('', cleaned)

        destination = (letter, int(num))

        promotion = PIECES.NONE
        meta_list = []

        if 'x' in move_str:
            takes = True
            target: PIECES = self.board.get_piece_at(destination)
            if is_white:
                meta_list.append("White takes {}.".format(target.value))
            else:
                meta_list.append("Black takes {}.".format(target.value))
        if '=' in move_str:  # promotion
            if piece == 'N':
                if is_white:
                    promotion = PIECES.WHITE_KNIGHT
                    meta_list.append("White pawn is promoted to {}.".format(promotion.value))
                else:
                    promotion = PIECES.BLACK_KNIGHT
                    meta_list.append("Black pawn is promoted to {}.".format(promotion.value))
            elif piece == 'B':
                if is_white:
                    promotion = PIECES.WHITE_BISHOP
                    meta_list.append("White pawn is promoted to {}.".format(promotion.value))
                else:
                    promotion = PIECES.BLACK_BISHOP
                    meta_list.append("Black pawn is promoted to {}.".format(promotion.value))
            elif piece == 'R':
                if is_white:
                    promotion = PIECES.WHITE_ROOK
                    meta_list.append("White pawn is promoted to {}.".format(promotion.value))
                else:
                    promotion = PIECES.BLACK_ROOK
                    meta_list.append("Black pawn is promoted to {}.".format(promotion.value))
            elif piece == 'Q':
                if is_white:
                    promotion = PIECES.WHITE_QUEEN
                    meta_list.append("White pawn is promoted to {}.".format(promotion.value))
                else:
                    promotion = PIECES.BLACK_QUEEN
                    meta_list.append("Black pawn is promoted to {}.".format(promotion.value))
            piece = ''  # keep as a pawn for move calculations
        if '++' in move_str:
            meta_list.append("Checkmate.")
        elif '+' in move_str:
            meta_list.append("Check.")
        if '#' in move_str:
            meta_list.append("Checkmate.")

        # look for the closest piece that follows the rules
        if piece == '':  # pawn
            p = PIECES.WHITE_PAWN
            if not is_white:
                p = PIECES.BLACK_PAWN
            source_loc = self.find_closest_piece(PIECES.PAWN, destination, is_white, takes=takes, hint=hint)
        elif piece == 'N':
            p = PIECES.WHITE_KNIGHT
            if not is_white:
                p = PIECES.BLACK_KNIGHT
            source_loc = self.find_closest_piece(PIECES.KNIGHT, destination, is_white, takes=takes, hint=hint)
        elif piece == 'B':
            p = PIECES.WHITE_BISHOP
            if not is_white:
                p = PIECES.BLACK_BISHOP
            source_loc = self.find_closest_piece(PIECES.BISHOP, destination, is_white, takes=takes, hint=hint)
        elif piece == 'R':
            p = PIECES.WHITE_ROOK
            if not is_white:
                p = PIECES.BLACK_ROOK
            source_loc = self.find_closest_piece(PIECES.ROOK, destination, is_white, takes=takes, hint=hint)
        elif piece == 'Q':
            p = PIECES.WHITE_QUEEN
            if not is_white:
                p = PIECES.BLACK_QUEEN
            source_loc = self.find_closest_piece(PIECES.QUEEN, destination, is_white, takes=takes, hint=hint)
        elif piece == 'K':
            p = PIECES.WHITE_KING
            if not is_white:
                p = PIECES.BLACK_KING
            source_loc = self.find_closest_piece(PIECES.KING, destination, is_white, takes=takes, hint=hint)

        if source_loc == (-1, -1):
            dprint("Move().evaluate_move: Problem with {} on move {}".format(move_str, p.value))
            self.has_problem = True


        if p != PIECES.NONE and source_loc != (-1, -1):
            self.set_expanded(source_loc, destination, p, is_white, meta_list)
            if promotion != PIECES.NONE:
                p = promotion
            self.board.set_piece_at(source_loc, PIECES.NONE)
            self.board.set_piece_at(destination, p)

    def set_val_dict(self, source: Tuple, destination: Tuple, piece: PIECES, is_white: bool):
        if is_white:
            self.white_val_dict["color"] = "white"
            self.white_val_dict["piece"] = piece.value.replace("white ","")
            self.white_val_dict["from"] = "{}{}".format(source[0], source[1])
            self.white_val_dict["to"] = "{}{}".format(destination[0], destination[1])
            self.white_val_dict["description"] = self.white_expanded
        else:
            self.black_val_dict["color"] = "black"
            self.black_val_dict["piece"] = piece.value.replace("black ","")
            self.black_val_dict["from"] = "{}{}".format(source[0], source[1])
            self.black_val_dict["to"] = "{}{}".format(destination[0], destination[1])
            self.black_val_dict["description"] = self.black_expanded


    def set_expanded(self, source: Tuple, destination: Tuple, piece: PIECES, is_white: bool, meta_list=None):
        if meta_list is None:
            meta_list = []
        text = "moves {} from {}{} to {}{}.".format(piece.value, source[0], source[1], destination[0], destination[1])

        for m in meta_list:
            text += " {}".format(m)
        if is_white:
            self.white_expanded = text
            self.white_expanded = "{} {}".format(self.white_player, text)
            if self.white_castles != CASTLING.NO:
                self.white_expanded = "{} {}.".format(self.white_player, self.white_castles.value)
        else:
            self.black_expanded = text
            self.black_expanded = "{} {}".format(self.black_player, text)
            if self.black_castles != CASTLING.NO:
                self.black_expanded = "{} {}.".format(self.black_player, self.black_castles.value)
        self.set_val_dict(source, destination, piece, is_white)

    def to_string(self) -> str:
        s = "raw: white: {}, black: {}\n".format(self.white_raw, self.black_raw)
        s += "expanded: \n\twhite: {}\n\tblack: {}\n".format(self.white_expanded, self.black_expanded)
        for c in self.comment_list:
            s += "\t{}\n".format(c)
        return s

    def build_query(self, move_num:int, d:Dict) ->str:
        keys = "(move_number"
        vals = "VALUES ({}".format(move_num)
        for key, val in d.items():
            keys += ", `{}`".format(key)
            vals += ', "{}"'.format(val)
        s = "INSERT into table_actual {}) {})".format(keys, vals)
        return s


    def to_sql(self, msi: MSI.MySqlInterface, move_num:int):
        # s = "raw: white: {}, black: {}\n".format(self.white_raw, self.black_raw)

        if 'unset' in self.white_expanded or 'unset' in self.black_expanded:
            dprint("Move.to_sql(): Problem with {} or {}".format(self.white_expanded, self.black_expanded))
            dprint(self.to_string())
        else:
            if 'castles' not in self.white_expanded:
                q = self.build_query(move_num, self.white_val_dict)
                print(q)
                msi.write_data(q)
            if 'castles' not in self.black_expanded:
                q = self.build_query(move_num, self.black_val_dict)
                msi.write_data(q)
                print(q)

class Game:
    event: str
    site: str
    match_date: datetime.date
    round: int
    white_player: str
    black_player: str
    result: str  # parse 1-0 = "white", 0-1 = "black", 1/2-1/2 = "draw"
    white_elo_rating: int
    black_elo_rating: int
    eco: str
    opening: str
    variation: str
    joined_lines: str
    move_list: List
    show_board_list: List
    file: TextIO
    meta_regex: Pattern
    move_regex: Pattern
    comment_regex: Pattern
    no_comment_regex: Pattern
    board: Chessboard
    end_of_game: bool
    has_problem: bool

    def __init__(self, f: TextIO):
        self.reset()
        self.parse_game(f)

    def reset(self):
        self.event = "unset"
        self.site = "unset"
        self.match_date: datetime.date.today()
        self.round = 0
        self.white_player = "unset"
        self.black_player = "unset"
        self.result = "unset"  # parse 1-0 = "white", 0-1 = "black", 1/2-1/2 = "draw"
        self.white_elo_rating = 0
        self.black_elo_rating = 0
        self.eco = "unset"
        self.opening = "unset"
        self.variation = "unset"
        self.joined_lines = ""
        self.move_list = []
        self.show_board_list = []
        self.meta_regex = re.compile('[^0-9a-zA-Z- .?,"]+')
        self.move_regex = re.compile('[\d]*(\.)')
        self.comment_regex = re.compile('[^{\}]+(?=})')
        self.no_comment_regex = re.compile('({[^}]*})')
        self.board = Chessboard()
        self.end_of_game = False
        self.has_problem = False

    def parse_meta(self, lines: List):
        key = lines[0].lower()
        val = lines[1].strip()
        if key == 'event':
            self.event = val
        if key == 'site':
            self.site = val
        if key == 'date':
            self.match_date: datetime.date
            try:
                self.match_date = datetime.datetime.strptime(val, '%Y.%m.%d')
            except ValueError:
                year = val.split('.')[0]
                self.match_date = datetime.datetime.strptime(year, '%Y')
        if key == 'round':
            try:
                self.round = int(val)
            except ValueError:
                self.round = -1

        if key == 'white':
            self.white_player = val
            vals = val.split(',')
            if len(vals) > 1:
                self.white_player = "{} {}".format(vals[1], vals[0]).strip()
        if key == 'black':
            self.black_player = val
            vals = val.split(',')
            if len(vals) > 1:
                self.black_player = "{} {}".format(vals[1], vals[0]).strip()
        if key == 'result':
            self.result = val
        if key == 'whiteelo':
            try:
                self.white_elo_rating = int(val)
            except ValueError:
                self.white_elo_rating = -1
        if key == 'blackelo':
            try:
                self.black_elo_rating = int(val)
            except ValueError:
                self.black_elo_rating = -1
        if key == 'eco':
            self.eco = val
        if key == 'opening':
            self.opening = val
        if key == 'variation':
            self.variation = val

    def parse_moves(self, line: str) -> bool:
        if len(line) > 2:
            self.joined_lines = "{} {}".format(self.joined_lines, line)
        if len(self.joined_lines) > 2 and len(line) < 2:
            # dprint("parse_moves '{}'".format(line))
            self.joined_lines = self.joined_lines.strip()
            line = self.move_regex.sub('xxx', self.joined_lines)
            line = line.lstrip('xxx')
            lines = line.split('xxx')
            # dprint("\tline = {}, lines = {}".format(line, lines))
            move_count = 1
            for l in lines:
                dprint("\nGame.parse_moves(): Move [{}] = '{}'".format(move_count, l))
                move_count += 1
                comment_list = self.comment_regex.findall(l)
                l = self.no_comment_regex.sub('', l)
                m = Move(l, self.white_player, self.black_player, self.board, comment_list)
                if move_count in self.show_board_list:
                    m.board.print_board(override=True)
                self.move_list.append(m)

                if m.has_problem or m.is_end:
                    self.has_problem = m.has_problem
                    return True
        return False

    def parse_line(self, line: str) -> bool:
        if "[" in line:
            line = self.meta_regex.sub('', line)
            lines = shlex.split(line)
            lines[1] = lines[1].replace('"', '')
            self.parse_meta(lines)
        else:
            is_last_move = self.parse_moves(line)
            if is_last_move:
                return True
        return False

    def parse_game(self, f: TextIO):
        done: bool = False
        while not done:
            line = f.readline()
            if line:
                done = self.parse_line(line.strip())
                if done:
                    self.end_of_game = True
                    # dprint("Found end of game {}".format(self.event))
            else:
                done = True

    def to_string(self) -> str:
        s = "\n--------------\nNo problem parsing: {} vs. {}, {}\n".format(self.white_player, self.black_player, self.match_date)
        if self.has_problem:
            s = "\n--------------\nProblem parsing game\n"
        s += "Event: {}\n".format(self.event)
        s += "Site: {}\n".format(self.site)
        s += "Date: {}\n".format(self.match_date)
        s += "Round: {}\n".format(self.round)
        s += "White: {}\n".format(self.white_player)
        s += "Black: {}\n".format(self.black_player)
        s += "Result: {}\n".format(self.result)
        s += "White rating: {}\n".format(self.white_elo_rating)
        s += "Black Rating: {}\n".format(self.black_elo_rating)
        s += "ECO: {}\n".format(self.eco)
        s += "---------------------------\nMoves:\n"

        count = 1
        for m in self.move_list:
            s += "[{}] {}\n".format(count, m.to_string())
            count += 1
        return s

    def to_sql(self, msi: MSI.MySqlInterface = None):
        m: Move
        count = 1
        for m in self.move_list:
            if random.random() < .4:
                m.to_sql(msi, count)
            count += 1
        # s += "raw: {}\n".format(self.joined_lines)


class PGNtoDB:
    input_filename: str
    msi: MSI.MySqlInterface

    def __init__(self, in_file: str, clear:bool=True):
        self.reset()
        self.input_filename = in_file
        self.parse_file()

    def reset(self):
        # dprint("PGNtoDB.reset")
        self.input_filename = "unset"
        self.msi = MSI.MySqlInterface("root", "postgres", "gpt2_chess")

    def parse_file(self):
        with open(self.input_filename) as f:
            done: bool = False
            while not done:
                g = Game(f)
                if g.end_of_game:
                    dprint(g.to_string())
                    if not g.has_problem:
                        g.to_sql(self.msi)
                        print(g.to_string())
                if not g.end_of_game:
                    done = True
        self.msi.close()

# 0 = off, 1 = text, 2 = board
print_state = 0
def main():

    # pte = PGNtoDB("../data/chess/VanWely.pgn", "../data/chess/narrative.txt")
    # pte = PGNtoDB("../data/chess/twic1329.pgn", "../data/chess/narrative.txt")
    # pte = PGNtoDB("../data/chess/OneTWIC.pgn")
    # pte = PGNtoDB("../data/chess/SixTwic.pgn",  "../data/chess/narrative.txt")
    file_list = glob.glob("../data/chess/*.pgn")
    for i in range(len(file_list)):
        print("Parsing{} reset = {}".format(file_list[i], i==0))
        PGNtoDB(file_list[i])


if __name__ == "__main__":
    main()
