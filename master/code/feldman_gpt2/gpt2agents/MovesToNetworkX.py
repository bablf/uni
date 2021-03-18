import gpt2agents.utils.MySqlInterface as MSI
import gpt2agents.utils.NetworkxGraphing as NG
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List


class MovesToNetworkX:
    msi:MSI.MySqlInterface
    ng:NG.NetworkxGraphing

    def __init__(self):
        self.reset()

    def reset(self):
        print("MovesToNetworkX.reset()")
        self.msi = MSI.MySqlInterface("root", "postgres", "gpt2_chess")
        self.ng = NG.NetworkxGraphing(name="chess1", creator="Phil")

    def build_from_query(self, q:str):
        result = self.msi.read_data(q)
        for d in result:
            print("row = {}".format(d))
            pd = {"piece": d['piece']}
            f = d['from']
            t = d['to']
            self.ng.add_connected_nodes(f, t, pd)

    def draw(self, do_show:bool = True):
        self.ng.draw(draw_legend=False, scalar=10, do_show=do_show)
        self.ng.print_stats()

    def find_closest_neighbors(self, depth:int=2, cur_depth:int = 1) -> List:
        return self.ng.find_closest_neighbors()

    def to_gefx(self, filename:str, graph_creator:str):
        self.ng.to_gexf(filename, graph_creator)

    def adjacency(self):
        filename = "../data/chess_adjacency_{}.xlsx".format(datetime.today().strftime('%Y-%m-%d_%H-%M'))
        self.ng.print_adjacency(filename)
        print()
        filename = "../data/chess_moves_{}.xlsx".format(datetime.today().strftime('%Y-%m-%d_%H-%M'))
        self.ng.print_edge_data(filename)

    def close(self):
        self.msi.close()

def main():
    save_graph = False

    mtn = MovesToNetworkX()
    #q = "select piece, `from`, `to` from table_moves where piece = 'pawn' or piece = 'king'  order by piece"
    q = "select piece, `from`, `to` from table_moves order by piece"
    mtn.build_from_query(q)
    mtn.adjacency()
    mtn.draw(do_show=False)
    mtn.close()

    knl = mtn.find_closest_neighbors()
    ng2 = NG.NetworkxGraphing(name="test2", creator="Phil")
    for n in knl:
        neighbor_list = n['known_nearest']
        name = n['node']
        for n2 in set(neighbor_list):
            ng2.add_connected_nodes(name, n2)
    ng2.draw(draw_legend=False, do_show=False, scalar=10)

    if save_graph:
        filename = "../data/chess_nearest_neighbors_{}.gml".format(datetime.today().strftime('%Y-%m-%d_%H-%M'))
        ng2.to_gml(filename, graph_creator="phil")

    plt.show()

if __name__ == "__main__":
    main()