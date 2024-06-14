# MCES4DG - # Maximum Common Edge Subgraph for Directed Graphs.
# We focus specifically on the Non-induced common subgraph.
# This is a heuristic algorithm that presumes that edges with greater arity anc most silimar arity are more likely to be mapped.

import HOGS2, HOGS
import sys, math
import time
import ShowGraphs
import networkx as nx
import random
# import numpy as np
# import pdb
import ShowGraphs
from grandiso import find_motifs
# import GrandisoComparison

global mode, term_separator
global max_topology_distance
global max_relational_distance
global numeric_offset
global G1, G2


max_topology_distance = 500  # in terms of a node's in/out degree
max_relational_distance = 0.99
numeric_offset = 1000
mode = "English"
# mode = 'Code'
if mode == "English":
    term_separator = "_"  # Map2Graphs.term_separator
else:
    term_separator = ":"


def generate_2_homomorphic_graphs_special():
    """ 2 Homomorphic Graphs for use by MCS edge non-induced """
    common_G = nx.MultiDiGraph()
    tripl_list1, tripli_list2, list1_paired_edges = [], [], []
    #common_G.add_edges_from([(1, 2), (2, 3), (2,3), (3,3), (3,3)])
    #common_G.add_edges_from([(2,7), (2,3), (2,10), (10,3), (2,6), (9,2), (9,9), (11,9), (11,8), (8,11), (11,5), (5,12), (0,4)])
    G1, G2 = nx.MultiDiGraph(), nx.MultiDiGraph()
    G1.add_edges_from( [(1, 8), (2, 8), (8, 1), (3, 7), (3, 13), (4, 15), (4, 15), (5, 1), (5, 3), (5, 7), (11, 6), (12, 9), (15, 0), (15, 4), (16, 4)] )
    G2.add_edges_from( [(1001, 1008), (1001, 1008), (1002, 1008), (1008, 1001), (1003, 1007), (1003, 1013), (1004, 1015), (1005, 1000),
     (1005, 1001), (1005, 1003), (1005, 1007), (1010, 1015), (1011, 1006), (1012, 1009), (1012, 1010), (1015, 1000), (1015, 1004), (1016, 1004)] )
    if False:
        G1 = common_G.copy()

        remapping = {}
        remapping.clear()
        for node in G1.nodes():
            remapping[node] = node + numeric_offset
        G2 = nx.relabel_nodes(G1, remapping, copy=True) #  duplicated the core graph.

    #G1.add_edge(3, 4) # then add some differences
    #G2.add_edge(1002, 1003)

    for a, b in G1.edges():
        tripl_list1.append([a, "to", b])
    for a, b in G2.edges():
        tripli_list2.append([a, "to", b])
    return G1, G2, tripl_list1, tripli_list2


def generate_2_homomorphic_graphs(graph_size = 55, prob=0.10, edge_deletions=0, node_deletions=0):
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

    tripl_list1, tripli_list2, list1_paired_edges = [], [], []
    remapping = {}

    remapping.clear()
    for node in G1.nodes():
        remapping[node] = node + numeric_offset
    G2 = nx.relabel_nodes(G1, remapping, copy=True)
    for a, b in G2.edges():
        tripli_list2.append([a, "to", b])
    for i in range(node_deletions):
        zz = G1.nodes(data=True)
        sorted_lis = sorted(list(zz), reverse=True)
        my_max = sorted_lis[0][0]
        indx = random.randint(0, my_max)
        G1.remove_node(indx)
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


def return_best_ismags_mapping_BACKUP(largest_common_subgraphs, t_encoding, t_decoding, G2):
    best_node_score, best_edge_score, best_node_map, best_pred_map = 0, 0, [], []
    for dic in largest_common_subgraphs:  # dict of alternative solutions
        dict_1 = list(dic.items())
        dict_1.sort(key=lambda i: i[0])
        node_map, pred_map = [], []  # node_map - mapped_edges, pred_map - list_of_mapped_preds_2
        for z in t_decoding.keys():
            node_map.append( (z, t_decoding[z]) )
        for n1,n2 in G2.edges():
            if n1 in t_encoding.keys() and n2 in t_encoding.keys():
                pred_map.append([(t_encoding[n1], t_encoding[n2]), (n1, n2) ])
        pred_map.sort(key=lambda i: i[0])

        nod_scor, edg_scor = score_numeric_mapping(pred_map)
        if edg_scor > best_edge_score:
            best_edge_score = edg_scor
            best_pred_map = pred_map
            best_node_map = node_map
            best_node_score = nod_scor
    return best_edge_score, best_node_map, best_pred_map


def return_best_ismags_mapping(largest_common_subgraphs, t_encoding, s_decoding, G1, G2):
    best_pred_map, best_edge_score, best_node_map, node_map, iter_n = [], 0, [], {}, 0
    largest_pred_map_size, best_edge_score, largest_node_map, largest_map_iter, best_score_iter = 0, 0, [], 0, 0
    largest_pred_map = []
    for this_mapping in largest_common_subgraphs:  # dict of alternative solutions
        pred_map = []  # node_map - mapped_edges, pred_map - list_of_mapped_preds_2
        node_map.clear()
        for n1,n2 in G2.edges():
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
        #    dud = 0
        nod_scor, edg_scor = score_numeric_mapping(pred_map)
        #if len(pred_map) > 0:
        #    print("ISMAGS PredMap", pred_map)
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


def score_numeric_mapping(pred_list):  # Using an edge-based metric
    """ Objective evaluation of randomly generated numeric graphs """
    global numeric_offset
    mapped_nodes, mismapped_nodes = set(), set()
    num_mapped_preds = 0
    if pred_list == []:
        return 0, 0
    if pred_list and len(pred_list[0]) == 3:
        for a,b,c in pred_list:  # includes edge label
            if a[0] == b[0] - numeric_offset:
                mapped_nodes.add(a[0])
            elif a[0] != b[0] - numeric_offset:
                mismapped_nodes.add(a[0])
            if a[2] == b[2] - numeric_offset:
                mapped_nodes.add(a[2])
            elif a[2] != b[2] - numeric_offset:
                mismapped_nodes.add(a[2])
            if a[0] == b[0] - numeric_offset and a[2] == b[2] - numeric_offset:
                num_mapped_preds += 1
    elif pred_list and len(pred_list[0]) == 2:  # without edge label
        for a,b in pred_list:
            if a[0] == b[0] - numeric_offset:
                mapped_nodes.add(a[0])
            elif a[0] != b[0] - numeric_offset:
                mismapped_nodes.add(a[0])
            if a[1] == b[1] - numeric_offset:
                mapped_nodes.add(a[1])
            elif a[1] != b[1] - numeric_offset:
                mismapped_nodes.add(a[1])
            if a[0] == b[0] - numeric_offset and a[1] == b[1] - numeric_offset:
                num_mapped_preds += 1
    return len(mapped_nodes), num_mapped_preds  # nodes & edges result
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
        elif G2_unmapped.has_edge(b[0], b[2]):
            G2_unmapped.remove_edge(b[0], b[2], label = a[1])
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
    G1_unmapped = G1.copy()
    G2_unmapped = G2.copy()
    for a,b in mapped_preds:  # 2 values, not 3
        counterpart_grf.add_edge( str(a[0])+" "+str(b[0]),  str(a[1])+" "+str(b[1])) #, label = str(a[1])+"-"+str(b[1]))
        if G1_unmapped.has_edge(a[0], a[1]):
            G1_unmapped.remove_edge(a[0], a[1])
        #else:
        #    G1_unmapped.remove_edge(a[0], a[2]) #, label = a[1])
        if b[1] == None:
            G2_unmapped.remove_edge(b[0], b[1])
        else:
            G2_unmapped.remove_edge(b[0], b[1]) #, label = a[1])
    G1_unmapped.remove_nodes_from(list(nx.isolates(G1_unmapped)))
    G2_unmapped.remove_nodes_from(list(nx.isolates(G2_unmapped)))
    return counterpart_grf, G1_unmapped, G2_unmapped


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
    """ comon, l1_only, l2_only = analyse_lists_of_paired_tuples(list1_paired_edges, pred_map_2) """
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
    for a in args:
        print(a, end=", ")
    print(" ")
    out_lis = []
    for a in args:
        out_lis.append(a )
    with open("Bigger-NodeDeletionResults.txt", "a") as file:
        file.write(str(out_lis) + "\n")
    #print("\n", name, ",T=,", round(time_diff, 2), ",Del=,", edge_deletions, ",N-Dels,", node_deletions,
    #      ",SIZE: Map'd Nods=,", round(len_mapping/max_map_nodes,2),
    #      len_mapping,"/", max_map_nodes, ", Edg=,", len_list_of_mapped_preds_1, ",/,", max_map_edges,
    #      round(len_list_of_mapped_preds_1/max_map_edges*100,2), ",%    CORRECT: Prft-Edg =,",
    #      round((scr_edges/max_map_edges), 2), ",%  Pfct-Nod=,", scr_nodes, ", Pfct-Pred=,", scr_edges,
    #      ",LCC,", len_largest_mapping_cc,",/,", len_largest_G1_cc,", ID,", grf_id)

#emit("this", "is" , "a", "test")
#emit("as", "is", "this", "is" , "a", "test")
#stop()

def graph_isomorphism_experiment(graph_size = 20, prob=0.10, edge_deletions=0, node_deletions=0):
    #import time
    #import ShowGraphs
    global G1, G2
    show_FDG = False
    list1_paired_edges = []
    ##################################### Generate 2 Homomorphic Graphs #################################
    G1, G2, tripli_list, tripli_list2 = generate_2_homomorphic_graphs(graph_size, prob, edge_deletions, node_deletions)
    #G1, G2, tripli_list, tripli_list_2 = generate_2_homomorphic_graphs_special()
    G1.graph['Graphid'] = str(graph_size) + " " + str(prob) + " " + str(edge_deletions) + " orig"
    G2.graph['Graphid'] = str(graph_size) + " " + str(prob) + " " + str(edge_deletions) + " vrnt"
    print("# Edges ", G1.number_of_edges(), "&", G2.number_of_edges(), end="      ")
    max_map_nodes = max( min(G1.number_of_nodes() - nx.number_of_isolates(G1),
                         G2.number_of_nodes() - nx.number_of_isolates(G2)), 1)
    max_map_edges = max(min(G1.number_of_edges(), G2.number_of_edges()), 1)  # avoid /0 error
    G1_preds, G2_preds = return_edges_as_triple_list(G1), return_edges_as_triple_list(G2)
    siz_g1, siz_g2 = len(G1_preds), len(G2_preds)
    if G1.number_of_edges() == 0 or G2.number_of_edges() == 0:
        print("  ### Empty Graph ### ")
        return
    grf_id = int((time.time() * 10000000) % 1000)

    if show_FDG and True:
        ShowGraphs.show_blended_space(G1_preds, [], [], \
            "Expt " + str(G1.number_of_nodes()) + " " + str(G1.number_of_edges()) + " " + str(grf_id) + " Tgt ")
        ShowGraphs.show_blended_space(G2_preds, [], [], \
            "Expt " + str(G1.number_of_nodes()) + " " + str(G1.number_of_edges()) + " " + str(grf_id) + " Src ")
    ################################## HOGS2 algorithm ###########################################
    time1 = time.time()
    list_of_mapped_preds_1, number_mapped_predicates, mapping = HOGS2.generate_and_explore_mapping_space(G1, G2, False)
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
            "Expt " + str(G1.number_of_nodes()) + " " + str(G1.number_of_edges()) + " " + str(grf_id) + " HOGS-2")
    emit("\nHOGS2, ", siz_g1, ",", siz_g2, ",T= , ", round(time2, 4), ", E-Dels= ,", edge_deletions, ",N-Dels,", node_deletions,
          ", SIZE: Map'd Nods= ,", round(len(mapping)/max_map_nodes,3), ", ", len(mapping)," , of , ", max_map_nodes,
          ", Edg= , ", round(len(list_of_mapped_preds_1)/max_map_edges*100,3), ",%,", len(list_of_mapped_preds_1), ", of , ", max_map_edges,
          ",     CORRECT: Prft-Edg= ,  ", round(scr_edges_pct, 3), ", %  Pfct-Nod= , ", scr_nodes, ", Pfct-Pred= , ", scr_edges,
          ", LCC , ",len(largest_mapping_cc), ", ", len(largest_G1_cc),",  ID , ", grf_id)

    # ############################################# VF2++ #############################################
    time1 = time.time()
    dic_res = nx.vf2pp_isomorphism(G1, G2)
    time_diff = time.time() - time1
    if dic_res:
        emit("VF2pp Isomorp, ", siz_g1, ",", siz_g2, ",T= , ", round(time_diff, 4), ", E-Dels= ,", edge_deletions,
              ",N-Dels,", node_deletions, " , nodes=", len(dic_res), " , ",
              dic_res)
    else:
        emit("VF2pp Failed, ", siz_g1, ",", siz_g2, ",T= , ", round(time_diff, 4), ", E-Dels= ,", edge_deletions,
              ",N-Dels,", node_deletions, " , nodes= 0")
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
    emit("DFS HOGS-1, ", siz_g1, ",", siz_g2, ",T= , ", round(time2, 4), ", E-Dels= ,", edge_deletions, ",N-Dels,", node_deletions,
          " , SIZE: Map'd Nods= , ", round(len(mapping)/max_map_nodes,3), " , ", len(mapping)," , of , ", max_map_nodes,
          " , Edg= , ", round(len(list_of_mapped_preds_1)/max_map_edges*100,3),",%,", len(list_of_mapped_preds_1), " , of , ", max_map_edges,
          ",    CORRECT: Prft-Edg = , ", round(scr_edges_pct, 3), " , % Pfct-Nod= , ", scr_nodes, " , Pfct-Pred= , ", scr_edges,
          " , LCC , ",len(largest_mapping_cc), " , of ", len(largest_G1_cc)," ,  ID , ", grf_id)
    # ######################################## Grandiso ##############################################
    time1 = time.time()
    # mapping_space = find_motifs(G1, G2)  # should work with Graph(), DiGraph()
    time2 = time.time() - time1
    best_scr, num_victors, best_Grandiso_mapping = 0, 0 ,{}
    # best_scr, num_victors, best_Grandiso_mapping = return_largest_Grandiso_mapping(mapping_space)
    list_of_mapped_preds_2 = return_grandiso_mapped_preds(best_Grandiso_mapping, G1, G2)
    scr_nodes, scr_edges = score_numeric_mapping(list_of_mapped_preds_2)
    counterpart_grf, G1_unmapped, G2_unmapped = generate_counterpart_graph(list_of_mapped_preds_1, G1, G2)
    scr_edges_pct = 100 * scr_edges / max_map_edges
    if counterpart_grf.number_of_nodes() > 0:
        largest_mapping_cc = max(nx.weakly_connected_components(counterpart_grf), key=len)
        largest_G1_cc = max(nx.weakly_connected_components(G1), key=len)
    else:
        largest_mapping_cc, largest_G1_cc = {}, {}
    emit("Grandiso, ", siz_g1, ",", siz_g2, ",T= , ", round(time2, 4),  ", E-Dels= ,", edge_deletions, ",N-Dels,", node_deletions,
          " , SIZE: Map'd Nods= , ", round(len(best_Grandiso_mapping)/max_map_nodes,3), " , ",
          len(best_Grandiso_mapping)," , of , ", max_map_nodes, " , Edg= , ", len(list_of_mapped_preds_1)," , of , ", max_map_edges, " , ",
          round(best_scr/max_map_edges*100,2), " , %   CORRECT: Prft-Edg = , ", round(scr_edges_pct, 2),
          " , %  Pfct-Nod= , ", scr_nodes, " , Pfct-Pred= , ", scr_edges,
          " , LCC , ",len(largest_mapping_cc), " , ", len(largest_G1_cc)," ,  ID , ", grf_id)
    return
    # ############################################# ISMAGS #############################################
    new_G1_ismags, s_encoding, s_decoding = encode_graph_labels(G1)
    new_G2_ismags, t_encoding, t_decoding = encode_graph_labels(G2)
    largest_common_subgraph = nx.MultiDiGraph()
    ismags = nx.isomorphism.ISMAGS(new_G1_ismags, new_G2_ismags)
    time1 = time.time()
    largest_common_subgraph = list(ismags.largest_common_subgraph(symmetry=False))
    time2 = time.time()
    scr_2, node_map_2, pred_map_2, largest_pred_map, largest_node_map, same_iter \
        = return_best_ismags_mapping(largest_common_subgraph, t_encoding, s_decoding, G1, G2)
    if type(largest_node_map) is not int:
        largest_node_map = 0
    scr_nodes, scr_edges = score_numeric_mapping(pred_map_2)
    if scr_edges > 0:
        counterpart_grf, G1_unmapped, G2_unmapped = generate_ismags_counterpart_graph(pred_map_2, G1, G2)
    elif len(largest_pred_map) > 0:
        counterpart_grf, G1_unmapped, G2_unmapped = generate_ismags_counterpart_graph(largest_pred_map, G1, G2)
    if counterpart_grf.number_of_edges() > 0:
        largest_mapping_cc = max(nx.weakly_connected_components(counterpart_grf), key=len)
    emit("ISMAGS, ", siz_g1, ",", siz_g2, ",T= , ", round(time2 - time1, 4),  ", E-Dels= ,", edge_deletions,
          ",N-Dels,", node_deletions,
          "  Map'D Nods= , ", round(largest_node_map/max_map_nodes,3),
          # " <<< ", largest_node_map , max_map_nodes, " >>> ",
          " , ", largest_node_map, " , of , ", max_map_nodes,
          " , Edg= , ", len(largest_pred_map), " , of , ", max_map_edges, " , ",
          round(len(largest_pred_map) / max_map_edges * 100, 2), ", of ,   CORRECT: , ",
          round(100 * scr_edges / max_map_edges, 2), " , % edges  Prft-Nod= , ", scr_nodes,
          " , P-Scor , ", scr_edges,
          " , LCC , ", len(largest_mapping_cc), " , ", len(largest_G1_cc), " ,  ID , ", grf_id,
          " , Same , ", same_iter, " ,   {", len(largest_common_subgraph), "} ,   ",)
    generic_preds = return_edges_as_triple_list(counterpart_grf)
    unmapped_G1_edges = return_edges_as_triple_list(G1_unmapped)
    unmapped_G2_edges = return_edges_as_triple_list(G2_unmapped)
    if show_FDG:
        ShowGraphs.show_blended_space(generic_preds, unmapped_G1_edges, unmapped_G2_edges,
                "Expt " + str(G1.number_of_nodes()) + " " + str(G1.number_of_edges()) + " " + str(grf_id) + " ISMAGS")
    print()
    #stop()

#######################################################################################################################
#######################################################################################################################

# def perform_igraph_experiments(G1, G2):
    # import igraph as ig
    #ig1 = ig.Graph.from_networkx(G1)
    #ig2 = ig.Graph.from_networkx(G2)

#######################################################################################################################
#######################################################################################################################


def run_graph_matching_tests():
    global G1, G2
    G1, G2 = nx.MultiDiGraph(), nx.MultiDiGraph()
    tot_edg_scr, loop_count = 0, 1
    for siz in range(25, 300, 5):  # ~ nodes ...25
        now_time = time.time()
        z = time.ctime(now_time)
        print("\n\n\n\#######", siz, "######################################### TIME:", z, end=" ")
        for prob in range(25, 151, 25): # 70, 136, 30   ...35
            prob1 = prob / 1000
            for dels in [0,1, 5, 30]: # range(1, 7, 2):   # dels=0 isomorphic graph matching, dels>0 homomorphic matching
                print("EXPT. Size, ", siz, "Prob. ", prob, "  EDg Delet", dels, "NodeDels", 1, end="  ")
                for repetition_n in range(5):  # reproducibility
                    graph_isomorphism_experiment(graph_size=siz, prob=prob1, edge_deletions=dels, node_deletions=1)
                    loop_count += 1
                    print()
                print("\n\n")
            # stop()
    print("\n\nOVERALL", tot_edg_scr/loop_count)
    # stop()


run_graph_matching_tests()
