import gpt2agents.utils.MySqlInterface as MSI
import pandas as pd
import re
from typing import List, Dict

class ChessStats:
    msi:MSI.MySqlInterface
    piece_list:List
    square_list:List
    color_list:List
    col_list:List
    row_list:List

    def __init__(self):
        self.reset()

    def reset(self):
        print("ChessStats.reset()")
        self.msi = MSI.MySqlInterface("root", "postgres", "gpt2_chess")
        self.color_list = ["black", "white"]
        self.piece_list = ["pawn", "rook", "knight", "bishop", "queen", "king"]
        self.col_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.row_list = [str(i) for i in range(1,9)]
        self.square_list = []
        for col in self.col_list:
            for row in self.row_list:
                self.square_list.append("{}{}".format(col, row))

    def get_count_result(self, count_query:str) -> int:
        r_list:List = self.msi.read_data(count_query)
        rd = r_list[0]
        val = rd['count(*)']
        return val

    def run_squares_query(self, table_name:str, constraint_str:str) -> Dict:
        d = {}
        prefix = 'select count(*) from {}'.format(table_name, constraint_str)
        total_moves = self.get_count_result("{};".format(prefix))
        for sq in self.square_list:
            print("square = {}".format(sq))
            rd = {"total_moves":total_moves}
            d[sq] = rd
            moves = self.get_count_result('{} where `from`= "{}" or `to` = "{}" {};'.format(prefix, sq, sq, constraint_str))
            rd["from_to_moves"] = moves
            for c in self.color_list:
                print("\tcolor = {}".format(c))
                moves = self.get_count_result('{} where (`from`= "{}" or `to` = "{}") and color = "{}" {};'.format(prefix, sq, sq, c, constraint_str))
                rd["{}_from_to_moves".format(c)] = moves
                moves = self.get_count_result('{} where `from`= "{}"  and color = "{}" {};'.format(prefix, sq, c, constraint_str))
                rd["{}_from_moves".format(c)] = moves
                moves = self.get_count_result('{} where `to`= "{}"  and color = "{}" {};'.format(prefix, sq, c, constraint_str))
                rd["{}_to_moves".format(c)] = moves
            for p in self.piece_list:
                print("\tpiece = {}".format(p))
                moves = self.get_count_result('{} where (`from`= "{}" or `to` = "{}") and piece = "{}" {};'.format(prefix, sq, sq, p, constraint_str))
                rd["{}_from_to_moves".format(p)] = moves
                moves = self.get_count_result('{} where `from`= "{}"  and piece = "{}" {};'.format(prefix, sq, p, constraint_str))
                rd["{}_from_moves".format(p)] = moves
                moves = self.get_count_result('{} where `to`= "{}"  and piece = "{}" {};'.format(prefix, sq, p, constraint_str))
                rd["{}_to_moves".format(p)] = moves
                for c in self.color_list:
                    for p in self.piece_list:
                        print("\tcolor = {}. piece = {}".format(c, p))
                        moves = self.get_count_result('{} where (`from`= "{}" or `to` = "{}") and piece = "{}" and color = "{}" {};'.format(prefix, sq, sq, p, c, constraint_str))
                        rd["{}_{}_from_to_moves".format(p, c)] = moves
                        moves = self.get_count_result('{} where `from`= "{}"  and piece = "{}" and color = "{}" {};'.format(prefix, sq, p, c, constraint_str))
                        rd["{}_{}_from_moves".format(p, c)] = moves
                        moves = self.get_count_result('{} where `to`= "{}"  and piece = "{}" and color = "{}" {};'.format(prefix, sq, p, c, constraint_str))
                        rd["{}_{}_to_moves".format(p, c)] = moves
        return d

    def get_square_indices(self, square:str) -> [int, int]:
        ci = self.col_list.index(square[0])
        ri = self.row_list.index(square[1])
        return ci, ri

    def calc_square_dist(self, sq1:str, sq2:str):
        c1, r1 = self.get_square_indices(sq1)
        c2, r2 = self.get_square_indices(sq2)
        return abs(c2-c1), abs(r2-r1)

    def run_legal_query(self, table_name:str, constraint_str:str) -> Dict:
        d = {}
        query = 'select count(*) from {};'.format(table_name, constraint_str)
        total_moves = self.get_count_result("{};".format(query))
        illegal_total = 0

        # pawns
        dd = {}
        d["pawns"] = dd
        query = 'select `from`, `to` from {} where piece = "pawn" {};'.format(table_name, constraint_str)
        illegal_moves = 0
        result_l = self.msi.read_data(query)
        for r in result_l:
            cd, rd = self.calc_square_dist(r["from"], r["to"])
            if cd > 1 or rd > 2:
                print("illegal pawn move: {}".format(r))
                illegal_moves += 1
        dd["illegal"] = illegal_moves
        query = 'select count(*) from {} where piece = "pawn" {};'.format(table_name, constraint_str)
        moves = self.get_count_result(query)
        dd["legal"] = moves - illegal_moves
        illegal_total += illegal_moves

        #rooks
        dd = {}
        d["rooks"] = dd
        query = 'select `from`, `to` from {} where piece = "rook" {};'.format(table_name, constraint_str)
        illegal_moves = 0
        result_l = self.msi.read_data(query)
        for r in result_l:
            cd, rd = self.calc_square_dist(r["from"], r["to"])
            if cd != 0 and rd != 0:
                print("illegal rook move: {}".format(r))
                illegal_moves += 1
        dd["illegal"] = illegal_moves
        query = 'select count(*) from {} where piece = "rook" {};'.format(table_name, constraint_str)
        moves = self.get_count_result(query)
        dd["legal"] = moves - illegal_moves
        illegal_total += illegal_moves

        # bishops
        dd = {}
        d["bishops"] = dd
        query = 'select `from`, `to` from {} where piece = "bishop" {};'.format(table_name, constraint_str)
        illegal_moves = 0
        result_l = self.msi.read_data(query)
        for r in result_l:
            cd, rd = self.calc_square_dist(r["from"], r["to"])
            if cd != rd:
                illegal_moves += 1
                print("illegal bishop move: {}".format(r))
        dd["illegal"] = illegal_moves
        query = 'select count(*) from {} where piece = "bishop" {};'.format(table_name, constraint_str)
        moves = self.get_count_result(query)
        dd["legal"] = moves - illegal_moves
        illegal_total += illegal_moves

        # knights
        dd = {}
        d["knights"] = dd
        query = 'select `from`, `to` from {} where piece = "knight" {};'.format(table_name, constraint_str)
        illegal_moves = 0
        result_l = self.msi.read_data(query)
        for r in result_l:
            cd, rd = self.calc_square_dist(r["from"], r["to"])
            if not ((cd == 2 and rd == 1) or (cd == 1 and rd == 2)) :
                print("illegal knight move: {}".format(r))
                illegal_moves += 1
        dd["illegal"] = illegal_moves
        query = 'select count(*) from {} where piece = "knight" {};'.format(table_name, constraint_str)
        moves = self.get_count_result(query)
        dd["legal"] = moves - illegal_moves
        illegal_total += illegal_moves

        #queen
        dd = {}
        d["queen"] = dd
        query = 'select `from`, `to` from {} where piece = "queen" {};'.format(table_name, constraint_str)
        illegal_moves = 0
        result_l = self.msi.read_data(query)
        for r in result_l:
            cd, rd = self.calc_square_dist(r["from"], r["to"])
            if not (cd == rd or cd == 0 or rd == 0):
                print("illegal queen move: {}".format(r))
                illegal_moves += 1
        dd["illegal"] = illegal_moves
        query = 'select count(*) from {} where piece = "queen" {};'.format(table_name, constraint_str)
        moves = self.get_count_result(query)
        dd["legal"] = moves - illegal_moves
        illegal_total += illegal_moves

        # king
        dd = {}
        d["king"] = dd
        query = 'select `from`, `to` from {} where piece = "king" {};'.format(table_name, constraint_str)
        illegal_moves = 0
        result_l = self.msi.read_data(query)
        for r in result_l:
            cd, rd = self.calc_square_dist(r["from"], r["to"])
            if cd > 1 or rd > 1:
                print("illegal king move: {}".format(r))
                illegal_moves += 1
        dd["illegal"] = illegal_moves
        query = 'select count(*) from {} where piece = "king" {};'.format(table_name, constraint_str)
        moves = self.get_count_result(query)
        dd["legal"] = moves - illegal_moves
        illegal_total += illegal_moves


        dd = {}
        d["totals"] = dd
        dd["illegal"] = illegal_total
        dd["legal"] = total_moves - illegal_total

        return d

    def to_spreadsheet(self, filename:str, d_dict:Dict):
        with pd.ExcelWriter(filename) as w:
            d:dict
            name:str
            for name, d in d_dict.items():
                df = pd.DataFrame.from_dict(d, orient='index')
                print("Dataframe: {}/{}\n{}".format(filename, name, df))
                df.to_excel(w, name)

    def to_string(self) -> str:
        s = "Colors: {}\nPieces: {}\nSquares: {}\nrows: {}\ncols: {}".format(
            self.color_list, self.piece_list, self.square_list, self.row_list, self.col_list)
        return s

def main():
    # constraint_str = 'and move_number < 42 and (description like "%White takes%" or description like "%Black takes%" or description like "%Check%")'
    constraint_str = ''
    cs = ChessStats()
    print(cs.to_string())
    table_name = "gpt_view_50"
    # ex_d = {}
    # ex_d["squares-table_moves"] = cs.run_squares_query(table_name, constraint_str)
    # ex_d["squares-table_actual"] = cs.run_squares_query("table_actual", constraint_str)
    # cs.to_spreadsheet("../results/stats_400.xlsx", ex_d)

    ex_d = {}
    ex_d["legal-table_moves"] = cs.run_legal_query(table_name, constraint_str)
    ex_d["legal-table_actual"] = cs.run_legal_query("table_actual", constraint_str)
    cs.to_spreadsheet("../results/legal_50.xlsx", ex_d)

if __name__ == "__main__":
    main()