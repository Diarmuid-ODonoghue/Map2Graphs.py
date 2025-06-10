# MCES4DG - # Maximum Common Edge Subgraph for Directed Graphs.
# We focus specifically on the Non-induced common subgraph.
# This is a heuristic algorithm that presumes that edges with greater arity anc most silimar arity are more likely
# to be mapped.

import HOGS2
import HOGS
# import vf3py
# import sys, math
import time
import networkx as nx
import random
import sys
# import pdb
import ShowGraphs
from grandiso import find_motifs
# import GrandisoComparison

global mode, term_separator
global max_topology_distance
global max_relational_distance
global numeric_offset
# global G1, G2


max_topology_distance = 500  # in terms of a node's in/out degree
max_relational_distance = 0.99
numeric_offset = 1000
mode = "English"
# mode = 'Code'
if mode == "English":
    term_separator = "_"  # Map2Graphs.term_separator
else:
    term_separator = ":"

################################################################################################
# https://github.com/betterenvi/Ullman-Isomorphism
import sys, collections, copy, re
import numpy as np

from Graph import *

class UllmanAlgorithm(object):
    """docstring for UllmanAlgorithm"""
    def __init__(self, *args, **kwargs):
        super(UllmanAlgorithm, self).__init__()

    def _init_params(self, g, q, display_M=False, display_mapping=True, max_num_iso=float('inf')):
        self.q = q # the query graph
        self.g = g # the large graph
        self.A = q.get_adjacency_matrix()
        self.B = g.get_adjacency_matrix()
        self.display_M = display_M
        self.display_mapping = display_mapping
        self.max_num_iso = max_num_iso
        self.done = 0 >= max_num_iso
        for attr in ['num_nodes', 'num_edges']:
            setattr(self, attr + '_q', getattr(q, attr))
            setattr(self, attr + '_g', getattr(g, attr))

        self._construct_M()
        self.avail_g = np.ones(self.num_nodes_g) # if one node in g is not used/mapped. the opposite of 'F' vector in the paper
        self.Ms_isomorphic = list()
        self.mappings = list()
        # for i, arg in enumerate(args):
        #     setattr(self, 'arg_' + str(i), arg)
        # for kw, arg in kwargs.items():
        #     setattr(self, kw, arg)

    def _construct_M(self):
        self.M = np.logical_and(self.q.node_labels.values[:, None] == self.g.node_labels.values,
            self.q.node_degrees.values[:, None] <= self.g.node_degrees.values)
        # the above code is equivalent to the following
        # self.M = np.zeros((self.num_nodes_q, self.num_nodes_g))
        # for i, nid_q in enumerate(self.q.nodelist):
        #     for j, nid_g in enumerate(self.g.nodelist):
        #         if self.q.node_labels[nid_q] == self.g.node_labels[nid_g] and self.q.degree(nid_q) <= self.g.degree(nid_g):
        #             self.M[i, j] = 1

        self.M = self.M.astype(int)
        self._refine_M(check_elabel=True)

    def _refine_M(self, max_iter=float('inf'), check_elabel=True):
        '''
        for any x, (A[i, x] == 1) ===> exist y s.t. M[x, y] == 1 and B[j, y] == 1 and elabel_q[i, x] == elabel_g[j, y]
        for any x, (A[i, x] == 1) ===> exist y s.t. (M[x, y] == 1 and BT[y, j] == 1 and el_q[i, x] == el_gT[y, j]). i.e.,
        for any x, (A[i, x] == 1) ===> (M[x, :] dot_prod (BT[:, j] col_ele_prod el_gT[:, j] == el_q[i, x])) >= 1 . i.e.,
        (A[i, :] == 1)  ===> diag(M dot_prod (BT[:, j] col_ele_prod (el_gT[:, j] outer_eq el_q[i, :]))) >= 1 . i.e.,
        A[i, :] <= diag(M dot_prod (BT[:, j] col_ele_prod (el_gT[:, j] outer_eq el_q[i, :]))).
        A[i, :] <= rowsum(M ele_prod ((el_qT[:, i] outer_eq el_g[j, :]) row_ele_prod B[j, :]))
        in numpy:
        A[i, :] <= (M * (el_q[:, i][:, None] == el_g[j, :]) * B[j, :]).sum(axis=1)

        if don't want to check edge label, then
        for any x, (A[i, x] == 1) ===> exist y s.t. (M[x, y] == 1 and B[j, y] == 1).
        A[i, :] <= rowsum(M ele_prod (E row_ele_prod B[j, :]))
        A[i, :] <= rowsum(M row_ele_prod B[j, :])  # in numpy: A[i, :] <= (M * E * B[j, :]).sum(axis=1)
        A[i, :] <= (M dot_prod BT[:, j]).
        Let Y = M dot_prod BT, then
        A[i, :] <= Y[:, j] = YT[j, :]

        this refinement process is iterative
        '''
        if check_elabel:
            el_g = self.g.edge_labels.values
            el_q = self.q.edge_labels.values
        changed = True
        num_iter = 0
        while changed and num_iter < max_iter:
            changed = False
            num_iter += 1
            for i in range(self.num_nodes_q):
                for j in range(self.num_nodes_g):
                    if check_elabel:
                        flag = self.M[i, j] > 0 and (self.A[i, :] > (self.M * (el_q[:, i][:, None] == el_g[j, :]) * self.B[j, :]).sum(axis=1)).any()
                    else:
                        flag = self.M[i, j] > 0 and (self.A[i, :] > self.M.dot(self.B.T[:, j])).any()
                    if flag:
                        self.M[i, j] = 0
                        changed = True

    def _check_isomorphic(self):
        '''
        check if (A[i, j] = 1) ==> (C[i, j] == 1) for any i, j
        '''
        C = self.M.dot((self.M.dot(self.B)).T)
        isomorphic = (self.A <= C).all()
        if isomorphic:
            self.Ms_isomorphic.append(copy.deepcopy(self.M))
            self.mappings.append(self._get_mapping(-1))
            self.done = len(self.mappings) >= self.max_num_iso
            if self.display_M:
                print (self.M)
            if self.display_mapping:
                print (self.mappings[-1])
        return isomorphic

    def _get_mapping(self, M_idx=None):
        if M_idx == None:
            res = list()
            for M_idx in range(len(self.Ms_isomorphic)):
                res.append(self._get_mapping(M_idx))
            return res
        elif M_idx >= len(self.Ms_isomorphic):
            return None
        M = self.Ms_isomorphic[M_idx]
        I, J = np.where(M == 1)
        return zip(list(I), list(J))

    def _dfs(self, depth=0):
        if self.done:
            return
        if depth >= self.num_nodes_q:
            self._check_isomorphic()
            return
        row = copy.deepcopy(self.M[depth, :])
        if (row * self.avail_g).sum() == 0:
            return
        self.M[depth, :] = 0
        for j in range(self.num_nodes_g):
            if row[j] == 1 and self.avail_g[j] == 1:
                self.M[depth, j] = 1
                self.avail_g[j] = 0
                self._dfs(depth + 1)
                self.avail_g[j] = 1
                self.M[depth, j] = 0
                if self.done:
                    break
        self.M[depth, :] = row

    def run(self, g, q, display_M=False, display_mapping=True, max_num_iso=float('inf')):
        '''
        max_num_iso: if max_num_iso isomorphic subgraphs have been found, then stop
        '''
        self._init_params(g, q, display_M=display_M, display_mapping=display_mapping, max_num_iso=max_num_iso)
        self._dfs(depth=0)

    def has_iso(self, g, q, display_M=False, display_mapping=True):
        '''
        check if g has at least 1 subgraph isomorphic to q
        '''
        self.run(g, q, display_M=display_M, display_mapping=display_mapping, max_num_iso=1)
        return len(self.mappings) > 0



def generate_2_homomorphic_graphs_special():
    """ 2 Homomorphic Graphs for use by MCS edge non-induced """
    tripl_list1, tripli_list2 = [], []
    # common_G.add_edges_from([(1, 2), (2, 3), (2,3), (3,3), (3,3)])
    # common_G.add_edges_from([(2,7), (2,3), (2,10), (10,3), (2,6), (9,2), (9,9), (11,9), (11,8), (8,11), (11,5),
    # (5,12), (0,4)])
    G1, G2, G_dud = nx.MultiDiGraph(), nx.MultiDiGraph(), nx.MultiDiGraph()
    G1.add_edges_from([(2, 10)])
    G2.add_edges_from([(1002, 1010), (1002, 1010), (1012, 1005)])
    #G1.add_edges_from([(0,2), (2, 7), (3, 11), (3, 16), (3, 18), (3, 22), (4, 2), (5, 21), (6, 2), (6, 22), (10, 15), (10, 21),
    # (11, 9), (12, 9), (12, 20), (14, 8), (14, 13), (15, 3), (18, 19), (19, 8), (20, 18), (20, 22), (21, 6), (24, 1)])
    #G1.add_node(99)
    #G1.add_edges_from([(1001, 1008), (1001, 1008), (1002, 1008), (1008, 1001), (1003, 1007), (1003, 1013),
    #    (1004, 1015), (1005, 1000), (9, 2), (9,2), (9,9),
    #    (1005, 1001), (1005, 1003), (1005, 1007), (1010, 1015), (1011, 1006), (1012, 1009),
    #    (1012, 1010), (1015, 1000), (1015, 1004), (1016, 1004)])
    #G1.add_edges_from([(1,2), (2,3), (3,4), (1,5), (6,7), (6,7), (10,11)])
    #G1.add_edges_from([(1001, 1008), (1001, 1008), (1002, 1008), (1,2), (1,2)])
    #G2 = G1.copy(as_view=False)
    if False:
        #G1 = common_G.copy()
        remapping = {}
        remapping.clear()
        for node in G1.nodes():
            remapping[node] = node + numeric_offset
        G2 = nx.relabel_nodes(G1, remapping, copy=True)  # duplicated the core graph.

    # G1.add_edge(3, 4) # then add some differences
    # G2.add_edge(1002, 1003)

    for a, b in G1.edges():
        tripl_list1.append([a, "to", b])
    for a, b in G2.edges():
        tripli_list2.append([a, "to", b])
    return G1, G2, tripl_list1, tripli_list2


def generate_2_homomorphic_graphs(graph_size=55, prob=0.10, edge_deletions=0, node_deletions=0):
    """ Generate 2 graphs, 1st with fewer edges than the second, forming Target and Source graphs. """
    global numeric_offset
    global G1, G2
    G_temp = nx.erdos_renyi_graph(graph_size, prob, directed=True)
    # FIXME Apparent error in ergos_renyi_graph, returning DiGraph instead of MultiDiGraph()
    din = list(d for n, d in G_temp.in_degree())
    dout = list(d for n, d in G_temp.out_degree())
    G1 = nx.directed_configuration_model(din, dout)
    # G1 = nx.scale_free_graph(graph_size, prob) #, 0.5, prob, directed=True)
    # G1.add_edges_from( [(0, 1), (1, 0), (1, 2), (2, 3)] ) # random_k_out_graph, newman_watts_strogatz_graph , ...
    if G1.number_of_edges() > 3000:    # Largest Graph Considered
        print("Graph might be too big, Return [] [] ")
        return nx.MultiDiGraph(), nx.MultiDiGraph(), [], []
    tripl_list1, tripli_list2 = [], []
    remapping = {}
    remapping.clear()
    for node in G1.nodes():
        remapping[node] = node + numeric_offset
    G2 = nx.relabel_nodes(G1, remapping, copy=True)
    for a, b in G2.edges():
        tripli_list2.append([a, "to", b])
    i = 0
    while i < node_deletions and G1.number_of_nodes() > 0:
        zz = G1.nodes(data=True)
        sorted_lis = sorted(list(zz), reverse=True)
        my_max = sorted_lis[0][0]
        indx = random.randint(0, my_max-1)
        if G1.has_node(indx):
            G1.remove_node(indx)
            i +=1
    for a, b in G1.edges():
        tripl_list1.append([a, "to", b])
    for i in range(edge_deletions):
        if len(tripl_list1) > 0:
            indx = random.randint(0, len(tripl_list1)-1)
            a, b = tripl_list1[indx][0], tripl_list1[indx][2]
            del(tripl_list1[indx])
            G1.remove_edge(a, b)
    return G1, G2, tripl_list1, tripli_list2

# ##########################################################################################################
# ############################################## MCS #######################################################
# ##########################################################################################################


def return_edges_as_triple_list(G, remove_none_rels = True):  # return_edges_as_triple_list(sourceGraph)
    """ returns a list of lists, each composed of triples"""
    res = []
    for (u, v, reln) in G.edges.data('label'):
        res.append([u, reln, v])
    return res
# return_edges_as_triple_list(targetGraph)


def return_best_ismags_mapping_BACKUP(largest_common_subgraphs, t_encoding, s_decoding, G1, G2):
    best_pred_map, best_edge_score, best_node_map, node_map, iter_n = [], 0, [], {}, 0
    largest_pred_map_size, best_edge_score, largest_node_map, largest_map_iter, best_score_iter = 0, 0, [], 0, 0
    largest_pred_map = []
    for this_mapping in largest_common_subgraphs:  # dict of alternative solutions
        pred_map = []  # node_map - mapped_edges, pred_map - list_of_mapped_preds_2
        node_map.clear()
        for n1, n2 in G2.edges():
            tn1_enc = t_encoding[n1]
            tn2_enc = t_encoding[n2]
            if tn1_enc in this_mapping.keys() and tn2_enc in this_mapping.keys():
                tn1_map, tn2_map = this_mapping[tn1_enc], this_mapping[tn2_enc]
                if tn1_map in s_decoding.keys() and tn2_map in s_decoding.keys():
                    tn1_map_decod, tn2_map_decod = s_decoding[tn1_map], s_decoding[tn2_map]
                    if (tn1_map_decod, tn2_map_decod) in G1.edges():
                        pred_map.append([(tn1_map_decod, tn2_map_decod), (n1, n2) ])
                        node_map[tn1_map_decod] = n1
                        node_map[tn2_map_decod] = n2
        nod_scor, edg_scor = score_numeric_mapping(pred_map)
        if edg_scor > best_edge_score:  # Best Scoring (CORRECT)
            best_edge_score = edg_scor
            best_pred_map = pred_map
            best_node_map.clear()
            best_node_map = node_map
            best_score_iter = iter_n
        if largest_pred_map_size < len(pred_map):  # Best size
            largest_pred_map_size = len(pred_map)
            largest_node_map = len(node_map)
            largest_pred_map = pred_map
            largest_map_iter = iter_n
    boleyn = best_score_iter == largest_map_iter and best_score_iter > 0
    return best_edge_score, best_node_map, best_pred_map, largest_pred_map, largest_node_map, boleyn


def return_best_ismags_mapping(largest_common_subgraphs, t_encoding, s_encoding, G1, G2):
    best_pred_map, best_edge_score, best_node_map, node_map, iter_n = [], 0, [], {}, 0
    largest_pred_map_size, best_edge_score, largest_node_map, largest_map_iter, best_score_iter = 0, 0, [], 0, 0
    largest_pred_map = []
    for this_mapping in largest_common_subgraphs:  # dict of alternative solutions
        pred_map = []  # node_map - mapped_edges, pred_map - list_of_mapped_preds_2
        node_map.clear()
        for n1, n2 in G1.edges():
            pairing_found = False
            if n1 in t_encoding.keys() and n2 in t_encoding.keys():
                n1_enc = t_encoding[n1]
                n2_enc = t_encoding[n2]
                for m1, m2 in G2.edges():
                    pairing_found = False
                    m1_enc = s_encoding[m1]
                    m2_enc = s_encoding[m2]
                    if n1_enc in this_mapping.keys() and n2_enc in this_mapping.keys() and \
                        m1_enc in this_mapping.values() and m2_enc in this_mapping.values():
                        pred_map.append([(n1, n2), (m1, m2)])
                        node_map[n1] = m1
                        node_map[n2] = m2
                        pairing_found = True
                        break
                if pairing_found:
                    continue  # skip to next G2 edge also
        """             if m1 == tn1_map_decod and m2 == tn2_map_decod:   """
        nod_scor, edg_scor = score_numeric_mapping(pred_map)
        if edg_scor > best_edge_score:  # Best Scoring (CORRECT)
            best_edge_score = edg_scor
            best_pred_map = pred_map
            best_node_map.clear()
            best_node_map = node_map
            best_score_iter = iter_n
        if len(pred_map) > largest_pred_map_size:  # Best size
            largest_pred_map_size = len(pred_map)
            largest_node_map = len(node_map)
            largest_pred_map = pred_map
            largest_map_iter = iter_n
    boleyn = best_score_iter == largest_map_iter and best_score_iter > 0
    return best_edge_score, best_node_map, best_pred_map, largest_pred_map, largest_node_map, boleyn


def score_numeric_mapping(pred_list):  # Using an edge-based metric
    """ Objective evaluation of randomly generated numeric graphs """
    global numeric_offset
    perfect_mapped_nodes, mismapped_nodes = set(), set()
    num_perfect_mapped_preds = 0
    if pred_list == []:
        return 0, 0
    if pred_list and len(pred_list[0]) == 3:  # includes edge label
        for a,b,c in pred_list:
            if a[0] == b[0] - numeric_offset:
                perfect_mapped_nodes.add(a[0])
            else:  # a[0] != b[0] - numeric_offset:
                mismapped_nodes.add(a[0])
            if a[2] == b[2] - numeric_offset:
                perfect_mapped_nodes.add(a[2])
            elif a[2] != b[2] - numeric_offset:
                mismapped_nodes.add(a[2])
            if a[0] == b[0] - numeric_offset and a[2] == b[2] - numeric_offset:
                num_perfect_mapped_preds += 1
    elif pred_list and len(pred_list[0]) == 2:  # without edge label
        for a,b in pred_list:
            if a[0] == b[0] - numeric_offset:
                perfect_mapped_nodes.add(a[0])
            elif a[0] != b[0] - numeric_offset:
                mismapped_nodes.add(a[0])
            if a[1] == b[1] - numeric_offset:
                perfect_mapped_nodes.add(a[1])
            elif a[1] != b[1] - numeric_offset:
                mismapped_nodes.add(a[1])
            if a[0] == b[0] - numeric_offset and a[1] == b[1] - numeric_offset:
                num_perfect_mapped_preds += 1
    return len(perfect_mapped_nodes), num_perfect_mapped_preds  # nodes & edges result
# print(score_numeric_mapping( [[(0, 1), (100, 101)], [(1, 2), (101, 102)], [(1, 2), (101, 102)]] ))


def generate_counterpart_graph(mapped_preds, G1, G2):
    counterpart_grf = nx.MultiDiGraph()
    G1_unmapped = G1.copy()
    G2_unmapped = G2.copy()
    for a,b,scr in mapped_preds:
        counterpart_grf.add_edge( str(a[0])+" "+str(b[0]),  str(a[2])+" "+str(b[2]), label = str(a[1])+"-"+str(b[1]))
        if a[1] == None and G1_unmapped.has_edge(a[0], a[2]):
            G1_unmapped.remove_edge(a[0], a[2])
        elif G1_unmapped.has_edge(a[0], a[2]):
            G1_unmapped.remove_edge(a[0], a[2], label = a[1])
        if b[1] == None and G2_unmapped.has_edge(b[0], b[2]):
            G2_unmapped.remove_edge(b[0], b[2])
        elif G2_unmapped.has_edge(b[0], b[2]):  # (b[0], b[2], {"label": b[1]})
            print("shit bag")
            print("shit bag")
            print("shit bag")
            print(" G2_unmapped.remove_edge(b[0], b[2], label = a[1]) ", b[1], "-", G2_unmapped.remove_edge(b[0], b[2], label = a[1]))
            G2_unmapped.remove_edge(b[0], b[2], label = b[1])
    G1_unmapped.remove_nodes_from(list(nx.isolates(G1_unmapped)))
    G2_unmapped.remove_nodes_from(list(nx.isolates(G2_unmapped)))
    return counterpart_grf, G1_unmapped, G2_unmapped


def return_largest_Grandiso_mapping(mapping_space):
    # print("Mapping space size:, ", len(mapping_space), end=" ")
    best_score, best_mapping, total_number_of_equal_best_mappings = 0, {}, 1
    for mapping in mapping_space:
        if len(mapping) > best_score:
            best_score = len(mapping)
            best_mapping = mapping
            total_number_of_equal_best_mappings = 1
        elif len(mapping) == best_score:
            total_number_of_equal_best_mappings += 1
    return best_score, total_number_of_equal_best_mappings, best_mapping


def return_grandiso_mapped_preds(best_Grand_map, G1, G2):
    """ Convert node pairs to a list of mapped predicate-pairs [[[p1_s p1_v p1_o], [p2_s p2_v p2_o]]  ...] """
    map_preds = []
    for (u, v, reln) in G1.edges.data('label'):
        if u in best_Grand_map.keys() and v in best_Grand_map.keys():
            map_preds.append([[u, reln, v], [best_Grand_map[u], reln, best_Grand_map[v]], 0.1])
    return map_preds


def generate_ismags_counterpart_graph(mapped_preds, G1, G2):
    """ Accepts lists of paired edges - without accompanying score"""
    counterpart_grf = nx.MultiDiGraph()
    G1_copy = G1.copy()
    G2_copy = G2.copy()
    for a,b in mapped_preds:  # 2 values, not 3
        counterpart_grf.add_edge( str(a[0])+" "+str(b[0]),  str(a[1])+" "+str(b[1])) #, label = str(a[1])+"-"+str(b[1]))
        if G1_copy.has_edge(a[0], a[1]):
            G1_copy.remove_edge(a[0], a[1])
        #else:
        #    G1_unmapped.remove_edge(a[0], a[2]) #, label = a[1])
        if G2_copy.has_edge(b[0], b[1]): #b[1] == None:
            G2_copy.remove_edge(b[0], b[1])
        #else:
        #    G2_copy.remove_edge(b[0], b[1]) #, label = a[1])
    G1_copy.remove_nodes_from(list(nx.isolates(G1_copy)))
    G2_copy.remove_nodes_from(list(nx.isolates(G2_copy)))
    return counterpart_grf, G1_copy, G2_copy


def encode_graph_labels(grf):
    nu_grf = nx.Graph()
    s_encoding = {}  # label, number
    s_decoding = {}  # number, label
    label = 0
    for x, y in grf.edges():
        if not x in s_encoding.keys():
            s_decoding[label] = x
            s_encoding[x] = label
            label += 1
        if not y in s_encoding.keys():
            s_decoding[label] = y
            s_encoding[y] = label
            label += 1
        nu_grf.add_edge(s_encoding[x], s_encoding[y])
    return nu_grf, s_encoding, s_decoding


def analyse_lists_of_paired_tuples(tup_list_1, tup_list_2):  # 2 lists of edges defining 2 graphs
    """ comon, l1_only, l2_only  """
    common = [[(a,b),(c,d)] for [(a,b),(c,d)] in tup_list_1 for [(p,q),(r,s)] in tup_list_2  if ((a==p) and (b==q) and (c==r)and (d==s))]
    list_1_only = tup_list_1.copy()
    list_2_only = tup_list_2.copy()
    list_1_only = [list_1_only.remove([(a,b),(c,d)]) for [(a,b),(c,d)] in common for [(p,q),(r,s)] in tup_list_1 if ((a==p) and (b==q) and (c==r)and (d==s))]
    list_2_only = [list_2_only.remove([(a,b),(c,d)]) for [(a,b),(c,d)] in common for [(p,q),(r,s)] in tup_list_2 if ((a==p) and (b==q) and (c==r)and (d==s))]
    return common, list_1_only, list_2_only


# ########################################################################################################
# ########################################################################################################
# ########################################################################################################

def emit(*args):
    """ print, putting comma between list items, and adding a newline at the end """
    for a in args:
        print(a, end=", ")
    print(" ")
    out_lis = []
    for a in args:
        out_lis.append(a )
    with open("C:/Users/dodonoghue/Documents/Python-Me/Cre8Blend/HOGS2-Evaluation.csv", "a") as file:
        file.write(str(out_lis) + "\n")
#emit("New File entry")


def graph_isomorphism_experiment_BACKUP(graph_size = 20, prob=0.10, edge_deletions=0, node_deletions=0):
    show_FDG = False
    ##################################### Generate 2 Homomorphic Graphs #################################
    G1, G2, tripli_list, tripli_list2 = generate_2_homomorphic_graphs(graph_size, prob, edge_deletions, node_deletions)
    # G1, G2, tripli_list, tripli_list_2 = generate_2_homomorphic_graphs_special()
    G1.graph['Graphid'] = str(graph_size) + " " + str(prob) + " " + str(edge_deletions) + " orig"
    G2.graph['Graphid'] = str(graph_size) + " " + str(prob) + " " + str(edge_deletions) + " vrnt"
    print("# Edges ", G1.number_of_edges(), "&", G2.number_of_edges(), end="      ")
    g1_num_mappable_nodes = G1.number_of_nodes() - nx.number_of_isolates(G1)
    g2_num_mappable_nodes = G2.number_of_nodes() - nx.number_of_isolates(G2)
    max_map_nodes = max( min(g1_num_mappable_nodes, g2_num_mappable_nodes), 1)  # avoid x/0 error
    max_map_edges = max(min(G1.number_of_edges(), G2.number_of_edges()), 1)  # avoid /0 error
    G1_preds, G2_preds = return_edges_as_triple_list(G1), return_edges_as_triple_list(G2)
    siz_g1, siz_g2 = len(G1_preds), len(G2_preds)
    if G1.number_of_edges() == 0 or G2.number_of_edges() == 0:
        print("  ### Empty Graph ### ")
        return
    grf_id = int((time.time() * 10000000) % 1000)
    #print("Isolates", nx.number_of_isolates(G1), nx.number_of_isolates(G2))
    if show_FDG and True:
        ShowGraphs.show_blended_space(G1_preds, [], [],
            "Expt " + str(G1.number_of_nodes()) + " " + str(G1.number_of_edges()) + " " + str(grf_id) + " Tgt ")
        ShowGraphs.show_blended_space(G2_preds, [], [],
            "Expt " + str(G1.number_of_nodes()) + " " + str(G1.number_of_edges()) + " " + str(grf_id) + " Src ")

    emit("Top level parameters","graph_size", graph_size, "prob", prob, "edge_deletions", edge_deletions,
         "node_deletions", node_deletions)
    ################################## HOGS2 algorithm ###########################################
    time1 = time.time()
    list_of_mapped_preds_1, number_mapped_predicates, mapping, relatio_structural_dist, rel_s2v, rel_count, \
        con_s2v, rel_count = HOGS2.generate_and_explore_mapping_space(G1, G2, False)
    time2 = time.time() - time1
    counterpart_grf, G1_unmapped, G2_unmapped = generate_counterpart_graph(list_of_mapped_preds_1, G1, G2)
    scr_nodes, scr_edges = score_numeric_mapping(list_of_mapped_preds_1)
    scr_edges_pct = 100 * scr_edges / max_map_edges
    largest_mapping_cc, largest_G1_cc = {}, {}
    if counterpart_grf.number_of_nodes() > 0:
        largest_mapping_cc = max(nx.weakly_connected_components(counterpart_grf), key=len)
        largest_G1_cc = max(nx.weakly_connected_components(G1), key=len)
    if show_FDG:
        generic_preds = return_edges_as_triple_list(counterpart_grf)
        unmapped_G1_edges = return_edges_as_triple_list(G1_unmapped)
        unmapped_G2_edges = return_edges_as_triple_list(G2_unmapped)
        ShowGraphs.show_blended_space_big_nodes(G1, generic_preds, unmapped_G1_edges, unmapped_G2_edges,
            "Expt " + str(G1.number_of_nodes()) + " " + str(G1.number_of_edges()) + " " + str(grf_id) + " HOGS-2")
    emit("\nHOGS2 ", "Nodes", g1_num_mappable_nodes, g2_num_mappable_nodes, "Edges=",
          siz_g1, siz_g2, "T= ", round(time2, 4), " E-Dels=", edge_deletions, " N-Dels= ", node_deletions,
          " SIZE: Map'd Nods= ", round(100*len(mapping)/max_map_nodes,3), len(mapping), " of ", max_map_nodes,
          " Edg=  ", round(len(list_of_mapped_preds_1)/max_map_edges*100,3), "%", len(list_of_mapped_preds_1), " of  ", max_map_edges,
          " CORRECT: Prft-Edg= ", round(scr_edges_pct, 3), " % Pfct-Nod= ", scr_nodes, " Pfct-Pred=  ", scr_edges,
          " LCC  ", len(largest_mapping_cc), "of", len(largest_G1_cc), "  ID  ", grf_id)
    #return
    # ############################ VF3 ############################
    #  VF3 requires NON-multi graphs.
    G1_undir = G1.to_undirected()
    G2_undir = G2.to_undirected()
    # ############################################# VF2++ #############################################
    time1 = time.time()
    dic_res = nx.vf2pp_isomorphism(G1, G2)
    time_diff = time.time() - time1
    if dic_res:
        emit("VF2pp Isomorp, ", siz_g1, siz_g2, " T= ", round(time_diff, 4), " E-Dels= ", edge_deletions,
              "N-Dels ", node_deletions, " nodes=", len(dic_res), dic_res)
    else:
        emit("VF2pp Failed ", "Nodes", G1.number_of_nodes(), G2.number_of_nodes(), siz_g1, siz_g2,
             " T= ", round(time_diff, 4), " E-Dels= ", edge_deletions, "N-Dels ", node_deletions, " nodes= 0")
    # return
    ################################## HOGS Original algorithm ###########################################
    time1 = time.time()
    list_of_mapped_preds_1, number_mapped_predicates, mapping = HOGS.generate_and_explore_mapping_space(G1, G2, False)
    time2 = time.time() - time1
    counterpart_grf, G1_unmapped, G2_unmapped = generate_counterpart_graph(list_of_mapped_preds_1, G1, G2)
    scr_nodes, scr_edges = score_numeric_mapping(list_of_mapped_preds_1)
    scr_edges_pct = 100 * scr_edges / max_map_edges
    largest_mapping_cc, largest_G1_cc = {}, {}
    if counterpart_grf.number_of_nodes() > 0:
        largest_mapping_cc = max(nx.weakly_connected_components(counterpart_grf), key=len)
        largest_G1_cc = max(nx.weakly_connected_components(G1), key=len)
    if show_FDG:
        generic_preds = return_edges_as_triple_list(counterpart_grf)
        unmapped_G1_edges = return_edges_as_triple_list(G1_unmapped)
        unmapped_G2_edges = return_edges_as_triple_list(G2_unmapped)
        ShowGraphs.show_blended_space_big_nodes(G1, generic_preds, unmapped_G1_edges, unmapped_G2_edges, \
            "Expt " + str(G1.number_of_nodes()) + " " + str(G1.number_of_edges()) + " " + str(grf_id) + " HOGS-1")
    emit("DFS HOGS-1 ", "Nodes", G1.number_of_nodes(), G2.number_of_nodes(), "Edges=",
          siz_g1, siz_g2, "T=", round(time2, 4), "E-Dels=", edge_deletions, "N-Dels", node_deletions,
          " SIZE: Map'd Nods=", round(len(mapping)/max_map_nodes,3), len(mapping), " of ", max_map_nodes,
          " Edg= ", round(len(list_of_mapped_preds_1)/max_map_edges*100,3), "%", len(list_of_mapped_preds_1), "of", max_map_edges,
          " CORRECT: Prft-Edg =", round(scr_edges_pct, 3), " % Pfct-Nod=", scr_nodes, " Pfct-Pred=", scr_edges,
          " LCC ", len(largest_mapping_cc), " of ", len(largest_G1_cc), " ID", grf_id)
    # return
    # ######################################## Grandiso ##############################################
    host = nx.fast_gnp_random_graph(7, 0.5)
    motif = nx.Graph()
    motif.add_edge("A", "B")
    motif.add_edge("B", "C")
    motif.add_edge("C", "D")
    motif.add_edge("D", "A")
    # mapping_space = find_motifs(host,host)
    # print("Grandiso RESULT", len(find_motifs(host, host)))
    # stop()
    G1_gi = nx.Graph()
    for n1,n2 in G1.edges():
        host.add_edge(n1, n2)
        break
    G2_gi = nx.Graph(G2)
    print("G1_gi.number_of_edges(), G2_gi.number_of_edges()", G1_gi.number_of_edges(), G2_gi.number_of_edges())
    max_map_edges_Grandiso = max(min(G1_gi.number_of_edges(), G2_gi.number_of_edges()),1)
    time1 = time.time()
    mapping_space = find_motifs(host, host) #, directed=True)  # should work with Graph(), DiGraph()
    time2 = time.time() - time1
    best_scr, num_victors, best_Grandiso_mapping = 0, 0 ,{}
    best_scr, num_victors, best_Grandiso_mapping = return_largest_Grandiso_mapping(mapping_space)
    list_of_mapped_preds_2 = return_grandiso_mapped_preds(best_Grandiso_mapping, G1, G2)
    scr_nodes, scr_edges = score_numeric_mapping(list_of_mapped_preds_2)
    counterpart_grf, G1_unmapped, G2_unmapped = generate_counterpart_graph(list_of_mapped_preds_1, G1, G2)
    scr_edges_pct = 100 * scr_edges / max_map_edges_Grandiso
    if counterpart_grf.number_of_nodes() > 0:
        largest_mapping_cc = max(nx.weakly_connected_components(counterpart_grf), key=len)
        largest_G1_cc = max(nx.weakly_connected_components(G1), key=len)
    else:
        largest_mapping_cc, largest_G1_cc = {}, {}
    emit("Grandiso ", siz_g1, siz_g2, "T=", round(time2, 4),  " E-Dels= ", edge_deletions, "N-Dels=", node_deletions,
          " SIZE: Map'd Nods= ", round(len(best_Grandiso_mapping)/max_map_nodes,3),
          len(best_Grandiso_mapping), " of ", max_map_nodes, "  Edg=", round(len(list_of_mapped_preds_1)/max_map_edges_Grandiso*100,3), "%",
         len(list_of_mapped_preds_1), " of ", max_map_edges,
          "CORRECT: Prft-Edg =", round(scr_edges_pct, 2),
          " %  Pfct-Nod= ", scr_nodes, "  Pfct-Pred= ", scr_edges,
          " LCC ", len(largest_mapping_cc), "of", len(largest_G1_cc), " ID ", grf_id)
    # ############################################# ISMAGS #############################################
    return
    new_G1_ismags, t_encoding, t_decoding = encode_graph_labels(G1)
    new_G2_ismags, s_encoding, s_decoding = encode_graph_labels(G2)
    # largest_common_subgraph = nx.MultiDiGraph()
    ismags = nx.isomorphism.ISMAGS(new_G1_ismags, new_G2_ismags)
    time1 = time.time()
    largest_common_subgraph = list(ismags.largest_common_subgraph(symmetry=False))
    time2 = time.time()
    #scr_2, node_map_2, pred_map_2, largest_pred_map, largest_node_map, same_iter \
    #    = return_best_ismags_mapping(largest_common_subgraph, s_encoding, t_decoding, G1, G2)
    scr_2, node_map_2, pred_map_2, largest_pred_map, largest_node_map, same_iter \
        = return_best_ismags_mapping(largest_common_subgraph, t_encoding, s_encoding, G1, G2)
    if type(largest_node_map) is not int:
        largest_node_map = 0
    scr_nodes, scr_edges = score_numeric_mapping(pred_map_2)
    if scr_edges > 0:
        counterpart_grf, G1_unmapped, G2_unmapped = generate_ismags_counterpart_graph(pred_map_2, G1, G2)
    elif len(largest_pred_map) > 0:
        counterpart_grf, G1_unmapped, G2_unmapped = generate_ismags_counterpart_graph(largest_pred_map, G1, G2)
    if counterpart_grf.number_of_edges() > 0:
        largest_mapping_cc = max(nx.weakly_connected_components(counterpart_grf), key=len)
    if len(largest_pred_map) > max_map_edges:
        print(" ISMAGS Error ")
        dud = 0
    emit("ISMAGS, ", g1_num_mappable_nodes, g2_num_mappable_nodes, "Edges=", siz_g1, siz_g2,
         " T=", round(time2 - time1, 4), " E-Dels=", edge_deletions, "N-Dels=", node_deletions,
          "  Map'D Nods=", round(largest_node_map/max_map_nodes,3),
          # " <<< ", largest_node_map , max_map_nodes, " >>> ",
          " , ", largest_node_map, " , of , ", max_map_nodes,
          " Edg=", len(largest_pred_map), " of ", max_map_edges,
          round(len(largest_pred_map) / max_map_edges * 100, 2), " of   CORRECT: , ",
          round(100 * scr_edges / max_map_edges, 2), " % edges  Prft-Nod=", scr_nodes,
          " P-Scor ", scr_edges, "  LCC  ", len(largest_mapping_cc),
          len(largest_G1_cc), "  ID ", grf_id, " Same ", same_iter, "  {", len(largest_common_subgraph), "}    ",)
    generic_preds = return_edges_as_triple_list(counterpart_grf)
    unmapped_G1_edges = return_edges_as_triple_list(G1_unmapped)
    unmapped_G2_edges = return_edges_as_triple_list(G2_unmapped)
    if show_FDG:
        ShowGraphs.show_blended_space(generic_preds, unmapped_G1_edges, unmapped_G2_edges,
                "Expt " + str(G1.number_of_nodes()) + " " + str(G1.number_of_edges()) + " " + str(grf_id) + " ISMAGS")
    print()
    #stop()



# ########################################################################################################
# ########################################################################################################
# ########################################################################################################

def return_standard_metrics(G1, G2):
    time1 = time.time()
    ged = nx.graph_edit_distance(G1, G2)  # Levenshtein for graphs
    time_diff1 = time.time() - time1
    time1 = time.time()
    opt_ged = nx.optimize_graph_edit_distance(G1, G2)
    time_diff2 = time.time() - time1
    time1 = time.time()
    oep = nx.optimize_edit_paths(G1, G2)
    time_diff3 = time.time() - time1
    if not type(opt_ged) == int:
        opt_ged = 0
    if not type(oep) == int:
        oep = 0
    # sematch and summar (textRank)
    #rim_rank = nx.simrank_similarity(G1, n1, n2)  # node similarity
    #panther_im = nx.panther_similarity(G1, n2, n2)
    return [["nx.graph_edit_distance", ged, time_diff1], ["nx.optimize_graph_edit_distance", opt_ged, time_diff2],
            ["nx.optimize_edit_paths(", oep, time_diff3]]

def node_match(n1, n2):
    if n1 == n2:
        return False
    elif int(n1) + 1000 == int(n2):
        return False
    else:
        return False

def node_subst_cost(n1, n2):
    if n1 == n2:
        return False
    elif int(n1) + 1000 == int(n2):
        return False
    else:
        return False


def graph_isomorphism_experiment(graph_size = 20, prob=0.10, edge_deletions=0, node_deletions=0):
    show_FDG = False
    ##################################### Generate 2 Homomorphic Graphs #################################
    G1, G2, tripli_list, tripli_list2 = generate_2_homomorphic_graphs(graph_size, prob, edge_deletions, node_deletions)
    # G1, G2, tripli_list, tripli_list_2 = generate_2_homomorphic_graphs_special()
    G1.graph['Graphid'] = str(graph_size) + " " + str(prob) + " " + str(edge_deletions) + " orig"
    G2.graph['Graphid'] = str(graph_size) + " " + str(prob) + " " + str(edge_deletions) + " vrnt"
    print("# Edges ", G1.number_of_edges(), "&", G2.number_of_edges(), end="      ")
    g1_num_mappable_nodes = G1.number_of_nodes() - nx.number_of_isolates(G1)
    g2_num_mappable_nodes = G2.number_of_nodes() - nx.number_of_isolates(G2)
    max_map_nodes = max( min(g1_num_mappable_nodes, g2_num_mappable_nodes), 1)  # avoid x/0 error
    max_map_edges = max(min(G1.number_of_edges(), G2.number_of_edges()), 1)  # avoid /0 error
    G1_preds, G2_preds = return_edges_as_triple_list(G1), return_edges_as_triple_list(G2)
    siz_g1, siz_g2 = len(G1_preds), len(G2_preds)
    if G1.number_of_edges() == 0 or G2.number_of_edges() == 0:
        print("  ### Empty Graph ### ")
        return
    grf_id = int((time.time() * 10000000) % 1000)
    #print("Isolates", nx.number_of_isolates(G1), nx.number_of_isolates(G2))
    if show_FDG and True:
        ShowGraphs.show_blended_space(G1_preds, [], [],
            "Expt " + str(G1.number_of_nodes()) + " " + str(G1.number_of_edges()) + " " + str(grf_id) + " Tgt ")
        ShowGraphs.show_blended_space(G2_preds, [], [],
            "Expt " + str(G1.number_of_nodes()) + " " + str(G1.number_of_edges()) + " " + str(grf_id) + " Src ")
    ################################## HOGS2 algorithm ###########################################
    time1 = time.time()
    list_of_mapped_preds_1, number_mapped_predicates, mapping, relatio_structural_dist, rel_s2v, rel_count, \
        con_s2v, rel_count = HOGS2.generate_and_explore_mapping_space(G1, G2, False)
    time2 = time.time() - time1
    counterpart_grf, G1_unmapped, G2_unmapped = generate_counterpart_graph(list_of_mapped_preds_1, G1, G2)
    scr_nodes, scr_edges = score_numeric_mapping(list_of_mapped_preds_1)
    scr_edges_pct = 100 * scr_edges / max_map_edges
    largest_mapping_cc, largest_G1_cc = {}, {}
    if counterpart_grf.number_of_nodes() > 0:
        largest_mapping_cc = max(nx.weakly_connected_components(counterpart_grf), key=len)
        largest_G1_cc = max(nx.weakly_connected_components(G1), key=len)
    if show_FDG:
        generic_preds = return_edges_as_triple_list(counterpart_grf)
        unmapped_G1_edges = return_edges_as_triple_list(G1_unmapped)
        unmapped_G2_edges = return_edges_as_triple_list(G2_unmapped)
        ShowGraphs.show_blended_space_big_nodes(G1, generic_preds, unmapped_G1_edges, unmapped_G2_edges,
            "Expt " + str(G1.number_of_nodes()) + " " + str(G1.number_of_edges()) + " " + str(grf_id) + " HOGS-2")
    emit("\nHOGS2 ", "Nodes", g1_num_mappable_nodes, g2_num_mappable_nodes, "Edges=",
          siz_g1, siz_g2, "T= ", round(time2, 4), " E-Dels=", edge_deletions, " N-Dels= ", node_deletions,
          " SIZE: Map'd Nods= ", round(100*len(mapping)/max_map_nodes,3), len(mapping), " of ", max_map_nodes,
          " Edg=  ", round(len(list_of_mapped_preds_1)/max_map_edges*100,3), "%", len(list_of_mapped_preds_1), " of  ", max_map_edges,
          " CORRECT: Prft-Edg= ", round(scr_edges_pct, 3), " % Pfct-Nod= ", scr_nodes, " Pfct-Pred=  ", scr_edges,
          " LCC  ", len(largest_mapping_cc), "of", len(largest_G1_cc), "  ID  ", grf_id)
    return
    # ############################ VF3 ############################
    #  VF3 requires NON-multi graphs.
    G1_undir = G1.to_undirected()
    G2_undir = G2.to_undirected()
    # ############################################# VF2++ #############################################
    time1 = time.time()
    dic_res = nx.vf2pp_isomorphism(G1, G2)
    time_diff = time.time() - time1
    if dic_res:
        emit("VF2pp Isomorp ", "Nodes", G1.number_of_nodes(), G2.number_of_nodes(), "Edges=", siz_g1, siz_g2,
             " T= ", round(time_diff, 4), " E-Dels= ", edge_deletions,
              "N-Dels ", node_deletions, " nodes=", len(dic_res), dic_res)
    else:
        emit("VF2pp Failed ", "Nodes=", G1.number_of_nodes(), G2.number_of_nodes(), "Edges=",siz_g1, siz_g2,
             " T= ", round(time_diff, 4), " E-Dels= ", edge_deletions, "N-Dels ", node_deletions, " nodes= 0")
    # return
    ################################## HOGS Original algorithm ###########################################
    time1 = time.time()
    list_of_mapped_preds_1, number_mapped_predicates, mapping = HOGS.generate_and_explore_mapping_space(G1, G2, False)
    time2 = time.time() - time1
    counterpart_grf, G1_unmapped, G2_unmapped = generate_counterpart_graph(list_of_mapped_preds_1, G1, G2)
    scr_nodes, scr_edges = score_numeric_mapping(list_of_mapped_preds_1)
    scr_edges_pct = 100 * scr_edges / max_map_edges
    largest_mapping_cc, largest_G1_cc = {}, {}
    if counterpart_grf.number_of_nodes() > 0:
        largest_mapping_cc = max(nx.weakly_connected_components(counterpart_grf), key=len)
        largest_G1_cc = max(nx.weakly_connected_components(G1), key=len)
    if show_FDG:
        generic_preds = return_edges_as_triple_list(counterpart_grf)
        unmapped_G1_edges = return_edges_as_triple_list(G1_unmapped)
        unmapped_G2_edges = return_edges_as_triple_list(G2_unmapped)
        ShowGraphs.show_blended_space_big_nodes(G1, generic_preds, unmapped_G1_edges, unmapped_G2_edges, \
            "Expt " + str(G1.number_of_nodes()) + " " + str(G1.number_of_edges()) + " " + str(grf_id) + " HOGS-1")
    emit("DFS HOGS-1 ", "Nodes", G1.number_of_nodes(), G2.number_of_nodes(), "Edges=",
          siz_g1, siz_g2, "T=", round(time2, 4), "E-Dels=", edge_deletions, "N-Dels", node_deletions,
          " SIZE: Map'd Nods=", round(len(mapping)/max_map_nodes,3), len(mapping), " of ", max_map_nodes,
          " Edg= ", round(len(list_of_mapped_preds_1)/max_map_edges*100,3), "%", len(list_of_mapped_preds_1), "of", max_map_edges,
          " CORRECT: Prft-Edg =", round(scr_edges_pct, 3), " % Pfct-Nod=", scr_nodes, " Pfct-Pred=", scr_edges,
          " LCC ", len(largest_mapping_cc), " of ", len(largest_G1_cc), " ID", grf_id)
    # ######################################## Grandiso ##############################################
    G1_gi, G2_gi = nx.DiGraph(), nx.DiGraph()  # non-Multi edges
    count, limit = 0, 50
    for n1, n2 in G1.edges():
        if G1_gi.has_edge(n1, n2):
            break
        G1_gi.add_edge(n1, n2)
        count += 1
    count, limit = 0, 50
    for n1, n2 in G2.edges():
        if G2_gi.has_edge(n1, n2):
            continue
        G2_gi.add_edge(n1, n2)
        count += 1
    max_map_edges_Grandiso = max(min(G1_gi.number_of_edges(), G2_gi.number_of_edges()), 1)
    #print("G1_gi.edges()", len(G1_gi.edges()), sorted(G1_gi.edges()))
    #print("G2_gi.edges()", len(G2_gi.edges()), sorted(G2_gi.edges()))
    if G1_gi.number_of_edges() > 0 and G2_gi.number_of_edges() > 0 and \
        nx.is_weakly_connected(G1_gi) and nx.is_weakly_connected(G2_gi):
        time1 = time.time()
        mapping_space = find_motifs(G2_gi, G1_gi) #, directed=True)  # (small ,in_big), MUST BE WEAKLY CONNECTED - should work with DiGraph()
        time2 = time.time() - time1
        best_scr, num_victors, best_Grandiso_mapping = return_largest_Grandiso_mapping(mapping_space)
    else:
        best_scr, num_victors, best_Grandiso_mapping = -1, -1, {}
    list_of_mapped_preds_2 = return_grandiso_mapped_preds(best_Grandiso_mapping, G1, G2)
    scr_nodes, scr_edges = score_numeric_mapping(list_of_mapped_preds_2)
    counterpart_grf, G1_unmapped, G2_unmapped = generate_counterpart_graph(list_of_mapped_preds_1, G1, G2)
    scr_edges_pct = 100 * scr_edges / max_map_edges
    if counterpart_grf.number_of_nodes() > 0:
        largest_mapping_cc = max(nx.weakly_connected_components(counterpart_grf), key=len)
        largest_G1_cc = max(nx.weakly_connected_components(G1), key=len)
    else:
        largest_mapping_cc, largest_G1_cc = {}, {}
    emit("Grandiso ", "Nodes", G1.number_of_nodes(), G2.number_of_nodes(), "Edges=",
          siz_g1, siz_g2,
         "T=", round(time2, 4),  "E-Dels= ", edge_deletions, "N-Dels=", node_deletions,
          " SIZE: Grd-Map'd Nods= ", round(len(best_Grandiso_mapping)/max_map_nodes,3),
          best_scr, " of ", max_map_nodes, "  Edg=", round(len(list_of_mapped_preds_2)/max_map_edges_Grandiso*100,3), "%",
         len(list_of_mapped_preds_2), " of ", max_map_edges_Grandiso,
          "CORRECT: Prft-Edg =", round(scr_edges_pct, 2), best_Grandiso_mapping,
          " %  Pfct-Nod= ", scr_nodes, "  Pfct-Pred= ", scr_edges,
          " LCC ", len(largest_mapping_cc), "of", len(largest_G1_cc), " ID ", grf_id,
          "reduced-data", G1_gi.number_of_nodes(), G2_gi.number_of_nodes(), G1_gi.number_of_edges(), G2_gi.number_of_edges())
    return
    # ############################################# GED #############################################
    time1 = time.time()
    ged_result = nx.graph_edit_distance(G1, G2) # node_subst_cost=0
    time2 = time.time() - time1
    emit("GrfEdtDist", "Nodes", siz_g1, siz_g2, "Edges=", siz_g1, siz_g2,
         "T=", round(time2, 4), " E-Dels= ", edge_deletions, "N-Dels=", node_deletions,
         "AverageSize", (siz_g1 + siz_g2)/2, " GED=", ged_result)
    # ############################################# O-GED #############################################
    time1 = time.time()  # should work for DiGraph()
    result = nx.optimize_graph_edit_distance(G1, G2)
    min_dist = sys.maxsize
    for v in result:                        # v is a dictionary # FIXME - this is not working properly
        #print("min_dist", min_dist)
        if v < min_dist:
            min_dist = v
    time2 = time.time() - time1
    emit("Optimize GrfEdtDist", "Nodes", siz_g1, siz_g2, "Edges=", siz_g1, siz_g2,
         "T=", round(time2, 4), "E-Dels=", edge_deletions, "N-Dels=", node_deletions, "Omz-GED=", min_dist)
    #emit("Omz-GED=", result)
    return
    # ############################################# GED #############################################
    time1 = time.time()  # should work for DiGraph()
    ged_result = nx.graph_edit_distance(G1_gi, G2_gi, node_subst_cost=node_subst_cost) # edge_del_cost
    time2 = time.time() - time1
    emit("NSC0-GrfEdtDist", "Nodes", siz_g1, siz_g2, "Edges=", siz_g1, siz_g2,
         "T=", round(time2, 4), " E-Dels= ", edge_deletions, "N-Dels=", node_deletions,
         "AverageSize", (siz_g1 + siz_g2)/2, " GED=", ged_result)
    # ############################################# O-GEP #############################################
    time1 = time.time()  # BRUTE FORCE ALGORITHM - slow but good.
    otml_gep_result, otml_ged_cost = nx.optimal_edit_paths(G1, G2)  # TODO very optimality of solutions 8 Nov '24
    time2 = time.time() - time1
    # lent = len(otml_ged_result[0])
    o_gep_best = 0
    for x in otml_gep_result:
        siz = 0
        for tup in x[1]:
            if None in tup:
                dud = 0
            else:
                siz += 1
        if siz > o_gep_best:
            o_gep_best = siz
        if o_gep_best > max_map_edges:
            dud = 0
    emit("Optimize GrfEdtPath", "Nodes", siz_g1, siz_g2, "Edges=", siz_g1, siz_g2,
         "T=", round(time2, 4), "E-Dels=", edge_deletions, "N-Dels=", node_deletions, "otml_ged_result=", o_gep_best)
    return
    ################################## Standard Metrics ###########################################
    for name, val, tim_dif in return_standard_metrics(G1, G2):
        scr_nodes, scr_edges = 0, 0
        break
        emit(name, "Nodes", g1_num_mappable_nodes, g2_num_mappable_nodes, "Edges=",
             siz_g1, siz_g2, "T= ", round(tim_dif, 4), " E-Dels=", edge_deletions, " N-Dels= ", node_deletions,
             " SIZE: Map'd Nods= ", round(100 * val / max_map_nodes, 3), val, " of ", max_map_nodes,
             " Edg=  ", round((len([]) / max_map_edges * 100, 3), "%", len([])),
             " of  ", max_map_edges,
             " CORRECT: Prft-Edg= ", round(0, 3), " % Pfct-Nod= ", "Unk", " Pfct-Pred=  ", scr_edges,
             " LCC  ", len([]), "of", len([]), "  ID  ", grf_id)
    # ############################################# ISMAGS #############################################
    new_G1_ismags, t_encoding, t_decoding = encode_graph_labels(G1)
    new_G2_ismags, s_encoding, s_decoding = encode_graph_labels(G2)
    # largest_common_subgraph = nx.MultiDiGraph()
    ismags = nx.isomorphism.ISMAGS(new_G1_ismags, new_G2_ismags)
    time1 = time.time()
    largest_common_subgraph = list(ismags.largest_common_subgraph(symmetry=False))
    time2 = time.time()
    #scr_2, node_map_2, pred_map_2, largest_pred_map, largest_node_map, same_iter \
    #    = return_best_ismags_mapping(largest_common_subgraph, s_encoding, t_decoding, G1, G2)
    scr_2, node_map_2, pred_map_2, largest_pred_map, largest_node_map, same_iter \
        = return_best_ismags_mapping(largest_common_subgraph, t_encoding, s_encoding, G1, G2)
    if type(largest_node_map) is not int:
        largest_node_map = 0
    scr_nodes, scr_edges = score_numeric_mapping(pred_map_2)
    if scr_edges > 0:
        counterpart_grf, G1_unmapped, G2_unmapped = generate_ismags_counterpart_graph(pred_map_2, G1, G2)
    elif len(largest_pred_map) > 0:
        counterpart_grf, G1_unmapped, G2_unmapped = generate_ismags_counterpart_graph(largest_pred_map, G1, G2)
    if counterpart_grf.number_of_edges() > 0:
        largest_mapping_cc = max(nx.weakly_connected_components(counterpart_grf), key=len)
    if len(largest_pred_map) > max_map_edges:
        print(" ISMAGS Error ")
        dud = 0
    emit("ISMAGS ", g1_num_mappable_nodes, g2_num_mappable_nodes, "Edges=", siz_g1, siz_g2,
         " T=", round(time2 - time1, 4), " E-Dels=", edge_deletions, "N-Dels=", node_deletions,
          "  Map'D Nods=", round(largest_node_map/max_map_nodes,3),
          # " <<< ", largest_node_map , max_map_nodes, " >>> ",
          largest_node_map, " of ", max_map_nodes,
          " Edg=", len(largest_pred_map), " of ", max_map_edges,
          round(len(largest_pred_map) / max_map_edges * 100, 2), " of   CORRECT: ",
          round(100 * scr_edges / max_map_edges, 2), " % edges  Prft-Nod=", scr_nodes,
          " P-Scor ", scr_edges, "  LCC  ", len(largest_mapping_cc),
          len(largest_G1_cc), "  ID ", grf_id, " Same ", same_iter, "  {", len(largest_common_subgraph), "}    ")
    generic_preds = return_edges_as_triple_list(counterpart_grf)
    unmapped_G1_edges = return_edges_as_triple_list(G1_unmapped)
    unmapped_G2_edges = return_edges_as_triple_list(G2_unmapped)
    if show_FDG:
        ShowGraphs.show_blended_space(generic_preds, unmapped_G1_edges, unmapped_G2_edges,
                "Expt " + str(G1.number_of_nodes()) + " " + str(G1.number_of_edges()) + " " + str(grf_id) + " ISMAGS")
    print()
    stop()


#######################################################################################################################
#######################################################################################################################

# def perform_igraph_experiments(G1, G2):
    # import igraph as ig
    #ig1 = ig.Graph.from_networkx(G1)
    #ig2 = ig.Graph.from_networkx(G2)

#######################################################################################################################
#######################################################################################################################


def run_graph_matching_tests():
    from datetime import datetime
    tot_edg_scr, loop_count = 0, 1
    for siz in range(50, 100, 25):  #range(35, 41, 5)
        print("\n#######", siz, "######################################### TIME:", end=" ")
        for prob in range(25, 76, 25):
            current_dateTime = datetime.now()
            print("TIME:", current_dateTime.hour, ":", current_dateTime.minute)
            prob1 = prob / 1000
            for node_dels in [0, 1, 2]: #, 5, 30]:  # dels=0 isomorphic, dels>0 homomorphic matching
                for edg_dels in [0, 1, 2]:
                    print("EXPT.  Size ", siz, "Prob. ", prob, "  Edg Dels", edg_dels, "NodeDels", node_dels)
                    emit("EXPT.  Size ", siz, "Prob. ", prob, "  Edg Dels", edg_dels, "NodeDels", node_dels)
                    for repetition_n in range(5):
                        graph_isomorphism_experiment(graph_size=siz, prob=prob1, edge_deletions=edg_dels, node_deletions=node_dels)
                        loop_count += 1
                        print()
            print("\n")
            #stop()
    print("\n\nOVERALL", tot_edg_scr/loop_count)


if False:
    g1, g2, g3, g4 = nx.MultiDiGraph(), nx.MultiDiGraph(), nx.MultiDiGraph(), nx.MultiDiGraph()
    #g1.add_edges_from([(1, 2), (2, 3), (3,2), (3,1)])
    #g2.add_edges_from([(11, 12), (12, 13), (13,12), (13,11)])
    g1_1 = nx.gnm_random_graph(4, 5)
    g1_2 = nx.gnm_random_graph(3, 4)
    g1 = g1_1.to_directed()
    g2 = g1_2.to_directed()
    res, num_mapped_preds, map_dict = [], 0, {}
    res, num_mapped_preds, map_dict = HOGS2.generate_and_explore_mapping_space(g1, g2, False)
    print(res, num_mapped_preds, map_dict, "\n")

    g3.add_edge("a", "b", label="e1")
    g3.add_edge("b", "c", label="e1")
    g3.add_edge("c", "a", label="e1")
    g4.add_edge("a2", "b2", label="e1")
    g4.add_edge("b2", "c2", label="e1")
    g4.add_edge("c2", "a2", label="e1")
    res, num_mapped_preds, map_dict = HOGS2.generate_and_explore_mapping_space(g3, g4, True)
    print(res, num_mapped_preds)
    stop()

run_graph_matching_tests()


