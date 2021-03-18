import networkx as nx
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter
from datetime import datetime
from typing import Dict, List

class NetworkxGraphing:
    node_size_dict:Dict
    edge_weight_list:List[List]
    G:nx.Graph
    name:str

    def __init__(self, name: str, creator: str = "none"):
        #  Create the graph
        self.reset()
        self.G = nx.Graph(name=name, creator=creator)
        self.name = name
        # print("creating {}".format(name))

    def reset(self):
        self.node_size_dict = {}
        self.edge_weight_list = [[]]
        self.G = None
        self.name = ""

    def add_connected_nodes(self, source:str, target:str, data_dict:Dict = {}):
        names = [source, target]
        for name in names:
            if name in self.node_size_dict.keys():
                self.node_size_dict[name] += 1
            else:
                # print("{}: adding node {}".format(self.name, name))
                self.node_size_dict[name] = 1
                self.G.add_node(name)

        nlist = nx.all_neighbors(self.G, source)
        if target not in nlist:
            self.G.add_edge(source, target, weight=0)
            self.G[source][target]['dict_array'] = []
        self.G[source][target]['weight'] += 1
        for key, val in data_dict.items():
            a:List = self.G[source][target]['dict_array']
            a.append({key:val})
        # print("{}: adding edge {} -- {}".format(self.name, source, target))

    def find_closest_neighbors(self, depth:int=2, cur_depth:int = 1) -> List:
        s_keys = sorted(list(nx.nodes(self.G)))
        all_nearest = []
        print("find_closest_neighbors(): nodes = {}".format(s_keys))
        for key in s_keys:
            #print("Testing neighbors of '{}' at depth {} of {}".format(key, cur_depth, depth))
            nlist = list(nx.all_neighbors(self.G, key))
            #print("\tneghbors = '{}'".format(nlist))
            # get neighbors for each element in the list
            # Note, this uses a Set, but a List could get counts of how many times a near neighbor occurs
            known_nearest = [] #set()
            for n in nlist:
                nlist2 = list(nx.all_neighbors(self.G, n))
                knl = list(set(nlist) & set(nlist2))
                if len(knl) > 0:
                    known_nearest.extend(knl) # How to add an element ot a list

            #all_nearest.append({"node":key, "known_nearest":list(known_nearest)})
            l = sorted(known_nearest)
            all_nearest.append({"node":key, "known_nearest":l})
        return all_nearest



    def draw(self, draw_legend:bool = True, font_size:int = 15, do_show:bool = True, scalar:float=100):
        f = plt.figure()

        # pos = nx.kamada_kawai_layout(self.G)
        pos=nx.spring_layout(self.G, iterations=500, seed=1) # positions for all nodes
        for key in self.node_size_dict.keys():
            carray = [[random.random()*0.5+0.5, random.random()*0.5+0.5, random.random()*0.5+0.5, 1.0]]
            size = int(self.node_size_dict[key])* scalar
            node_list = [key]
            nx.draw_networkx_nodes(self.G, pos,
                                   nodelist= node_list,
                                   node_color= carray,
                                   node_size=size,
                                   label=key)

        #  Draw edges and labels using defaults
        nx.draw_networkx_edges(self.G,pos)
        nx.draw_networkx_labels(self.G,pos, font_size=font_size)

        #  Render to pyplot
        # How to add a legend: https://stackoverflow.com/questions/22992009/legend-in-python-networkx
        #plt.gca().legend('one', 'two', 'three', 'four')
        if draw_legend:
            plt.legend(loc='right')
        if do_show:
            plt.show()

    def print_adjacency(self, file_name:str = None):
        s_keys = sorted(list(nx.nodes(self.G)))
        print("print_adjacency(): nodes = {}".format(s_keys))
        z = np.zeros((len(s_keys), len(s_keys))).astype(int)

        df = pd.DataFrame(z, index=s_keys, columns=s_keys,)
        for key in s_keys:
            # df[key][key] = -1 # show the diagonal
            n = nx.all_neighbors(self.G, key)
            for node in n:
                edges = nx.edges(self.G, [node])
                df[key][node] = self.G[key][node]['weight']
                # print("[{}][{}] = {}".format(key, node, self.G[key][node]['dict_array']))
                #df[key][node] += 1
                #df[key][node] += int(self.node_size_dict[key])

        if file_name != None:
            df.to_excel(file_name, index=True, header=True)
        print(df)

    def print_edge_data(self, filename:str = None):
        s_keys = sorted(list(nx.nodes(self.G)))
        print("print_edge_data(): nodes = {}".format(s_keys))
        for key in s_keys:
            n = nx.all_neighbors(self.G, key)
            for node in n:
                print("[{}][{}]: Weight = {} data = = {}".format(key, node, self.G[key][node]['weight'], self.G[key][node]['dict_array']))

        if filename != None:
            print("creating workbook {}".format(filename))
            workbook = xlsxwriter.Workbook(filename)
            worksheet = workbook.add_worksheet(name="moves")
            row = 0
            columns = ['square1', 'square2', 'weight', 'piece', 'piece', 'piece', 'piece', 'piece', 'piece', 'piece']
            for i in range(len(columns)):
                worksheet.write(row, i, columns[i])

            for key in s_keys:
                n = nx.all_neighbors(self.G, key)
                for node in n:
                    row += 1
                    worksheet.write(row, 0, key)
                    worksheet.write(row, 1, node)
                    worksheet.write(row, 2, self.G[key][node]['weight'])
                    da = self.G[key][node]['dict_array']
                    for i in range(len(da)):
                        d:Dict = da[i]
                        worksheet.write(row, i+3, d['piece'])
            workbook.close()
            print("finished workbook {}".format(filename))

    def to_simple_graph(self, graph_creator:str) -> nx.Graph:
        G = nx.Graph(name=self.G.name, creator=graph_creator)
        s_keys = sorted(set(nx.nodes(self.G)))
        for key in s_keys:
            G.add_node(key)

        for key in s_keys:
            nlist = list(nx.all_neighbors(self.G, key))
            for n in nlist:
                G.add_edge(key, n)
        return G

    def to_gml(self, filename:str, graph_creator:str):
        G = self.to_simple_graph(graph_creator)
        nx.write_gml(G, filename)

    def to_graphml(self, filename:str, graph_creator:str):
        G = self.to_simple_graph(graph_creator)
        nx.write_graphml_lxml(G, filename)

    def to_gexf(self, filename:str, graph_creator:str):
        # we have to create a new graph without the dict entry
        G = self.to_simple_graph(graph_creator)
        nx.write_gexf(G, filename)

    def print_stats(self):
        print("Graph '{}':".format(self.name))
        print("\tG.graph = {0}".format(self.G.graph))
        print("\tG.number_of_nodes() = {0}".format(self.G.number_of_nodes()))
        print("\tG.number_of_edges() = {0}".format(self.G.number_of_edges()))

def stats(ng:NetworkxGraphing):
    ng.print_adjacency()
    print()
    ng.print_edge_data()
    print()


    ng.print_stats()

    ng.draw(draw_legend=False, do_show=False)

if __name__ == '__main__':
    random.seed(1)
    ng = NetworkxGraphing(name="test", creator="Phil")

    letters = ['a', 'b', 'c']
    nodes = []
    for c in letters:
        for i in range(3):
            nodes.append("{}{}".format(c, i+1))
    pieces = ['pawn', 'rook', 'bishop', 'knight', 'queen', 'king']
    for i in range(30):
        source = random.choice(nodes)
        target = random.choice(nodes)
        if source != target:
            ng.add_connected_nodes(source, target, {"piece": random.choice(pieces)})
    knl = ng.find_closest_neighbors()
    stats(ng)
    for n in knl:
        print(n)

    filename = "../data/chess_all_neighbors_{}.gexf".format(datetime.today().strftime('%Y-%m-%d_%H-%M'))
    #ng.to_gexf(filename, graph_creator="phil")
    print("\n----------------------\n")


    ng2 = NetworkxGraphing(name="test2", creator="Phil")
    for n in knl:
        neighbor_list = n['known_nearest']
        name = n['node']
        for n2 in neighbor_list:
            ng2.add_connected_nodes(name, n2)
    stats(ng2)
    filename = "../data/chess_nearest_neighbors_{}.gexf".format(datetime.today().strftime('%Y-%m-%d_%H-%M'))
    #ng2.to_gexf(filename, graph_creator="phil")

    plt.show()

