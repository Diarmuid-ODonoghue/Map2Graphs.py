 # HOGS is a heuristic algorithm that presumes that edges with greater arity
# should be given a greater priority than edges with a smaller arity.
# It explores the edge-space of graph-subgraph near isomorphism.
# loosely inspired by Nawaz, Enscore and Ham (NEH) algorithm
# local optimisation, near the global optimum.
# Edges are described by a 4-tuple of in/out degrees from a di-graph. 2 edges are compared by Wasserstein metric.
# I believe it's an admissible heuristic! A narrow search space is explored heuristically.
# Homomorphic Erdos-Renyi Graph Search
# Subgraph Subgraph Isomorphism Erdos-Renyi graph Search (HOGS - SSIS)
# Can I use a simple if statement to skip over the second and subsequent edges on the target graph during search?
# Homomorphic Graph Degree guided Search HGDGS
# edge_align()

import sys
import math
import networkx as nx
import numpy as np
# import tqdm

global mode
global term_separator
global max_topology_distance
global semantic_threshold
global max_relational_distance
global numeric_offset
global tgt_edge_vector_dict
global src_edge_vector_dict
global target_number_of_nodes
global target_number_of_edges
global source_number_of_nodes
global source_number_of_edges
global wnl


max_topology_distance = 20  # in terms of a node's in/out degree
numeric_offset = 1000
beam_size = 4  # beam breadth for beam search
epsilon = 100
current_best_mapping = []
bestEverPredMapping = []
bestEverNodeMapping = {}


mode = "English"
mode = 'Code'
if mode == "English":
    term_separator = "_"  # Map2Graphs.term_separator
    max_conceptual_distance = 0.4999
    max_relational_distance = 0.4999
    semantic_threshold = 0.9  # MAXimum semantic distance allowed.
    if True:
        from nltk.stem import WordNetLemmatizer
        wnl = WordNetLemmatizer()
else:
    term_separator = ":"
    max_conceptual_distance = 0.88

global s2v
if False: # Not need for source code operations
    from sense2vec import Sense2Vec
    # s2v = Sense2Vec().from_disk("C:/Users/user/Documents/Python-Me/Sense2Vec/s2v_reddit_2019_lg/")
    s2v = Sense2Vec().from_disk("C:/Users/dodonoghue/Documents/Python-Me/Sense2Vec/s2v_reddit_2019_lg/")
    query = "drive|VERB"


global s2v_verb_cache
s2v_verb_cache = dict()
s2v_verb_cache = {'abc': 0}  # to avoid a null key problem
global s2v_noun_cache
s2v_noun_cache = dict()
s2v_noun_cache = {'abc': 0}  # to avoid a null key problem
# print(s2v_verb_cache['abc'])
# query = "can|VERB"
# assert query in s2v


def find_nearest(vec):
    for key, vec in s2v.items():
        print(key, vec)


def MultiDiGraphMatcher(target_graph, souce_graph):
    generate_and_explore_mapping_space(target_graph, souce_graph, False)


def is_isomorphic():
    print(" HOGS DFS.is_isomorphic() ")

def get_freq(word):
    return s2v.get_freq(word)

def similarity(w1, w2):
    #print("XX", w1, w2)
    if get_freq(w1) is None or get_freq(w2) is None:
        return 1
    else:
        return s2v.similarity(w1, w2)



# @staticmethod
def generate_and_explore_mapping_space(target_graph, source_graph, semantics=True,
                                       identical_edges_nodes=True):  # new one
    global current_best_mapping, bestEverPredMapping, semantic_threshold, epsilon
    global tgt_edge_vector_dict, src_edge_vector_dict, beam_size
    global target_number_of_nodes, target_number_of_edges
    global source_number_of_nodes, source_number_of_edges
    global mode, max_relational_distance
    if identical_edges_nodes:
        mode = "Code"
        max_relational_distance = 0.05  # edges only
        max_conceptual_distance = 0.05  # nodes only
    else:
        mode = "English"
    target_number_of_nodes = target_graph.number_of_nodes()
    target_number_of_edges = target_graph.number_of_edges()
    source_number_of_nodes = source_graph.number_of_nodes()
    source_number_of_edges = source_graph.number_of_edges()
    current_best_mapping, bestEverPredMapping = [], []
    tgt_edge_vector_dict, src_edge_vector_dict = {}, {}
    if target_graph.number_of_edges() == 0 or source_graph.number_of_edges() == 0:
        return [], 0, {}
    uniq_target_pairs = trim_multi_edges(target_graph)   # separate iteration over self-edges
    uniq_source_pairs = trim_multi_edges(source_graph)
    # Selects ONE SAMPLE relation in each direction
    ordered_target_preds, tgt_edge_vector_dict = return_sorted_predicates(target_graph, uniq_target_pairs)  # search only over reduced space with
    ordered_source_preds, src_edge_vector_dict = return_sorted_predicates(source_graph, uniq_source_pairs)
    ordered_candidate_sources = []
    bestEverPredMapping, mapping, relatio_structural_dist, rel_s2v, rel_count, con_s2v, con_count = [], [], \
        0.0, 0.0, 0, 0.0, 0
    for sum_t_vec, t_subj, t_obj in ordered_target_preds:  # start from most highly referenced nodes
        best_distance, composite_distance, best_subj, best_reln, best_obj \
                          = sys.maxsize, sys.maxsize, "nil", "nil", "nil"
        alternate_candidates, alternates_confirmed = [], []
        tgt_vect = tgt_edge_vector_dict[t_subj, t_obj]
        t_relns = return_edges_between_these_objects(t_subj, t_obj, target_graph)
        for sum_s_vec, s_subj, s_obj in ordered_source_preds:
            src_vect = src_edge_vector_dict[s_subj, s_obj]
            diff_vect = compare_vectors(tgt_vect, src_vect)
            s_relns = return_edges_between_these_objects(s_subj, s_obj, source_graph)
            topology_dist = np.sqrt(sum(diff_vect))  # rms_topology_dist
            if False and topology_dist > max_topology_distance:
                print(" MAX topol dist", end="")
                continue
            elif ((t_subj == t_obj) and (s_subj != s_obj)) or ((s_subj == s_obj) and (t_subj != t_obj)):
                continue   # if one is a self-map && the other not ... then no match
            if semantics:
                s_relns = return_edges_between_these_objects(s_subj, s_obj, source_graph)
                reslt_lis = align_relations_single_arity_only(t_relns, s_relns, semantics)  # reslt_lis[0] is best
                reln_dist = reslt_lis[0][0]  # most similar relations - relational_distance(t_reln, s_reln)
                if reln_dist > max_relational_distance:  # or subj_dist > semantic_threshold or obj_dist > semantic_threshold:
                    continue
                # TODO: Check specific subsection/s of diff_vect for excessive dissimilarity
                subj_dist = conceptual_distance(t_subj, s_subj)
                obj_dist = conceptual_distance(t_obj, s_obj)
            else:
                reln_dist, subj_dist, obj_dist, t_reln, s_reln = 0, 0, 0, None, None
            combo_dist = scoot_ahead(t_subj, t_relns[0], t_obj, s_subj, s_relns[0], s_obj, source_graph, target_graph, semantics)
            # h_prime = combo_dist # math.sqrt(combo_dist)  # level=1
            # print(reln_dist,  subj_dist , obj_dist, topology_dist)
            composite_distance = ((reln_dist + subj_dist + obj_dist)*2.1) + (topology_dist) #+ (h_prime)
            #composite_distance = (reln_dist*5 + subj_dist + obj_dist) + topology_dist/2 + h_prime/2
            if composite_distance < best_distance:         # minimize distance
                best_distance = composite_distance
            alternate_candidates.append([t_subj, t_obj, s_subj, s_obj, composite_distance])
        alternate_candidates.sort(key=lambda x: x[4])  # sort by composite distance
        if len(alternate_candidates) > 0:
            alternates_confirmed = []
            for x in alternate_candidates:
                if abs(x[4] - best_distance) < epsilon: # and best_distance < 250.00
                    alternates_confirmed.append(x)  # flat list of sublists
        alternates_confirmed = alternates_confirmed[:beam_size]  # consider BEST options only, Threshold?
        ordered_candidate_sources.append(alternates_confirmed)  # ordered_candidate_sources
    reslt = explore_mapping_space(target_graph, source_graph, ordered_target_preds, ordered_candidate_sources,
                                      [], semantics)
    relatio_structural_dist, rel_s2v, rel_count, con_s2v, con_count = \
        evaluate_mapping(target_graph, source_graph, reslt, semantics)  # bestEverPredMapping
    print("SCORES-HOGS2", relatio_structural_dist, " ", rel_s2v, rel_count, con_s2v, rel_count)
    dud = 0
    #if target_graph.graph['Graphid'] == "98 Ratterman E3 2 MereApp.T.RVB":
    print("***")
    print('Graphid',  target_graph.graph['Graphid'] )
    return bestEverPredMapping, len(bestEverPredMapping), mapping, relatio_structural_dist, rel_s2v, rel_count, \
        con_s2v, con_count


def trim_multi_edges(grf):  # remove Multi edges from iterable search space
    """ [[1, 3, 3, 0, 3, 1, 'hunter_he', 'pledged', 'hawk'] ..."""
    visited_pairs = []
    edge_list = grf.edges.data("label")
    for x,y,r in edge_list:  # use wordnet lemmas count()
        if [x, y] in visited_pairs:  # one relation in each direction
            pass
        else:
            visited_pairs.append([x,y])
    return visited_pairs

#################################################################################################################
######################################### Scoot Ahead ###########################################################
#################################################################################################################


def return_best_in_combo(targetGraph, sourceGraph, tgt_subj, src_subj, tgt_preds, src_preds, semantics=True):
    """ Given: tgt_subj, src_subj; Select best incoming Node-and-Edge combination. Guided search by topological
       similarity.         {'lake': {0: {'label': 'fed_by'}}, 'they': {0: {'label': 'conveyed'}}}) """
    global tgt_edge_vector_dict, src_edge_vector_dict
    in_t_rel_list, in_s_rel_list = [], []
    for in_tgt_nbr, node in tgt_preds:  # find MOST similar incoming S & T pair
        if tgt_subj != node:
            continue
        elif not (in_tgt_nbr, tgt_subj) in tgt_edge_vector_dict:
            print("Missing edge vector from tgt_edge_vector_dict")
            continue
        tgt_vect = tgt_edge_vector_dict[(in_tgt_nbr, tgt_subj)]
        if semantics:
            temp = list(node)[0]
            if isinstance(temp, int):  # irregularity arising from NetworkX merge_nodes()
                rel_n = node[0]['label']
                in_t_rel_list.append([in_tgt_nbr, tgt_subj, tgt_vect, rel_n])
            else:
                in_t_rel_list.append([in_tgt_nbr, tgt_vect, temp])
        else:
            in_t_rel_list.append([in_tgt_nbr, tgt_vect, None])
    for in_src_nbr, node in src_preds:
        if src_subj != node:
            continue
        elif not (in_src_nbr, src_subj) in src_edge_vector_dict:
            print("Missing edge vector from src_edge_vector_dict")
            continue  # TODO:
        src_vect = src_edge_vector_dict[in_src_nbr, src_subj]
        if semantics:
            temp = list(node)[0]
            if isinstance(temp, int):  # irregularity arising from NetworkX merge_nodes()
                rel_n = node[0]['label']
                in_s_rel_list.append([in_src_nbr, src_vect, rel_n])
            else:
                in_s_rel_list.append([in_src_nbr, src_vect, temp])
        else:
            in_s_rel_list.append([in_src_nbr, src_vect, None])
    if in_t_rel_list == [] and in_s_rel_list == []:
        return [], 0
    elif in_t_rel_list == []:
        return [], sum(src_vect)
    elif in_s_rel_list == []:
        return [], sum(tgt_vect)
    reslt = align_edges_single_arity_only(in_t_rel_list, in_s_rel_list, semantics)
    if reslt == []:
        scr = 0
    else:
        scr = reslt[0][0]
    return reslt, scr


def return_best_out_combo(targetGraph, sourceGraph, tgt_obj, src_obj, t_preds, s_preds, semantics=True):
    """{'lake': {0: {'label': 'fed_by'}}, 'they': {0: {'label': 'conveyed'}}}) """
    global tgt_edge_vector_dict, src_edge_vector_dict
    out_t_rel_list, out_s_rel_list = [], []
    if len(list(t_preds)) == 0 or len(list(s_preds)) == 0:
        return [], 0
    for out_tgt_nbr, reln in t_preds:
        tgt_vect = tgt_edge_vector_dict[(tgt_obj, out_tgt_nbr)]
        if semantics:
            tmp = list(reln)[0]
            if isinstance(tmp, int):  # Required because of irregularity arising from merging nodes
                zz = reln[0]['label']  # XXX
                out_t_rel_list.append([out_tgt_nbr, tgt_vect, zz])
            else:
                out_t_rel_list.append([out_tgt_nbr, tgt_vect, tmp])
        else:
            out_t_rel_list.append([out_tgt_nbr, tgt_vect, None])
    for out_s_nbr, foovalue2 in s_preds:
        src_vect = src_edge_vector_dict[(src_obj, out_s_nbr)]
        if semantics:
            tmp = list(foovalue2)[0]
            if isinstance(tmp, int):  # Required because of irregularity arising from merging nodes NetworkX
                zz = foovalue2[0]['label']  # XXX
                out_s_rel_list.append([out_s_nbr, src_vect, zz])
            else:
                out_s_rel_list.append([out_s_nbr, src_vect, tmp])
        else:
            out_s_rel_list.append([out_s_nbr, src_vect, None])

    reslt = align_edges_single_arity_only(out_t_rel_list, out_s_rel_list, semantics)   # Now
    if reslt == []:
        scr = 0
    else:
        scr = reslt[0][0]
    return reslt, scr


def align_relations_single_arity_only(t_rels_list, s_rels_list, semantics):
    """ Find single best incoming/outgoing link alignment. t_rels_list = node, vector, relation
        Used by scoot_ahead to estimate maximal mapping between incoming links to an edge's Agnt role.
        Returns the single best solution - best tgt -> src relation mapping."""
    global max_relational_distance
    result_list, rel_dist = [], 1
    for t_rel in t_rels_list:
        for s_rel in s_rels_list:
            if semantics:
                rel_dist = max(relational_distance(t_rel, s_rel), 0.001)
                if rel_dist > max_relational_distance:
                    rel_dist = 50.5  # FIXME set to 1000 say?
            #diff_vect = compare_vectors(t_vec, s_vec)
            #topo_dist = max( np.sqrt(sum(diff_vect)), 0.001)
            #prod_dist = rel_dist * topo_dist
            result_list.append([rel_dist, None, t_rel, None, s_rel])
    rslt = sorted(result_list, key=lambda x:x[0])
    return rslt  # from align_edges_single_arity_only

def align_edges_single_arity_only(t_rels_list, s_rels_list, semantics=False):
    """ Find single best incoming/outgoing link alignment. t_rels_list = node, vector, relation
        Used by scoot_ahead to estimate maximal mapping between incoming links to an edge's Agnt role.
        Returns the single best solution - best tgt -> src relation mapping. """
    global max_relational_distance
    result_list, rel_dist = [], 1
    for t_nn, t_vec, t_rel in t_rels_list:  #  for t_rel in t_rels_list:
        for s_nn, s_vec, s_rel in s_rels_list:  # for s_rel in s_rels_list:
            if t_rel != None:  # semantics:
                rel_dist = max(relational_distance(t_rel, s_rel), 0.001)
                if rel_dist > max_relational_distance:
                    rel_dist = 50.5  # FIXME set to 1000 say?
            diff_vect = compare_vectors(t_vec, s_vec)
            topo_dist = max( np.sqrt(sum(diff_vect)), 0.001)
            prod_dist = rel_dist * topo_dist
            result_list.append([prod_dist, t_nn, t_rel, s_nn, s_rel])
    rslt = sorted(result_list, key=lambda x:x[0])
    return rslt

def compare_vectors(vec1, vec2):
    return (vec1 - vec2) ** 2  # consider np.sqrt(sum(


def scoot_ahead(t_subj, t_reln, t_obj, s_subj, s_reln,  s_obj, sourceGraph, targetGraph, semantics):
    if type(sourceGraph).__name__ == "MultiDiGraph" or type(sourceGraph).__name__ == "DiGraph":
        best_in_links, in_rel_sim = return_best_in_combo(targetGraph, sourceGraph, t_subj, s_subj,
                                       targetGraph.in_edges(t_subj), sourceGraph.in_edges(s_subj), semantics)
        best_out_links, out_rel_sim = return_best_out_combo(targetGraph, sourceGraph, t_obj, s_obj,
                                        targetGraph.succ[t_obj].items(), sourceGraph.succ[s_obj].items(), semantics)
    else:     # FIXME Cater for NON-Directed Graphs too :-)
        best_in_links, in_rel_sim = return_best_in_combo(targetGraph, sourceGraph, t_subj, s_subj,
                                       targetGraph.edges(t_subj), sourceGraph.edges(s_subj), semantics)
        best_out_links, out_rel_sim = return_best_out_combo(targetGraph, sourceGraph, t_obj, s_obj,
                                        targetGraph.succ[t_obj].items(), sourceGraph.succ[s_obj].items(), semantics)
    combined_distance = in_rel_sim + out_rel_sim  # reln_dist + in_rel_sim
    if type(best_in_links) is list and type(best_out_links) is list:
        if len(best_in_links) > 0 and len(best_out_links) > 0:
            if len(best_in_links[0])>2 and len(best_out_links[0])>2:
                combined_distance = best_in_links[0][0] + best_out_links[0][0]
    # FIXME get-degree (in-node t- out-node)
    return combined_distance


def dist_to_sim(dis):
    return (dis - 1) * -1


def second_head(node):
    global term_separator
    if not (isinstance(node, str)):
        return ""
    else:
        lis = node.split(term_separator)
        if len(lis) >= 2:
            wrd = lis[1].strip()
        else:
            wrd = lis[0].strip()
        return wrd


# #####################################################################################################################
# #####################################################################################################################
# #####################################################################################################################


def explore_mapping_space(t_grf, s_grf, t_preds_list, s_preds_list, globl_mapped_predicates, semantics=True):
    """ Map the next target t_pred, by finding a mapping from the recommended sources in s_preds_list.
    Solution is complied recursively, adding one from t_preds_list and one from s_preds_list at a time.
    bestEverPredMapping stores the best ever recorded solution. """
    #if len(s_preds_list) <2:
    #    dud = 0
    global target_number_of_nodes, target_number_of_edges
    global source_number_of_nodes, source_number_of_edges
    global max_topology_distance, bestEverPredMapping, bestEverNodeMapping
    global tgt_edge_vector_dict, src_edge_vector_dict
    if len(globl_mapped_predicates) > len(bestEverPredMapping):  # compare scores, not lengths?
        bestEverPredMapping = globl_mapped_predicates
        bestEverNodeMapping.clear()
        bestEverNodeMapping = return_dict_from_mapping_list(bestEverPredMapping)
    if t_preds_list == [] or s_preds_list == []:
        return bestEverPredMapping  # global_mapped_predicates
    if len(t_preds_list) + len(globl_mapped_predicates) < len(bestEverPredMapping):  # abandon early
        return bestEverPredMapping
    elif len(t_preds_list) + len(globl_mapped_predicates) < len(bestEverPredMapping):  # abandon early
        print("Finhish Early 1")
        return globl_mapped_predicates
    #elif len(globl_mapped_predicates) + math.floor(len(t_preds_list)) < len(bestEverNodeMapping):  # FIXME more running totals needed
    #    dud = 0
    #    print("Finhish Early 2", t_preds_list, " ---- ", s_preds_list)
    #    return bestEverPredMapping
    if s_preds_list[0] == [] or t_preds_list[0] == []:
        return explore_mapping_space(t_grf, s_grf, t_preds_list[1:], s_preds_list[1:], globl_mapped_predicates, semantics)
    elif t_preds_list != [] and s_preds_list[0] == []:
        explore_mapping_space(t_grf, s_grf, t_preds_list, s_preds_list[1:], globl_mapped_predicates, semantics)
        # return globl_mapped_predicates
    t_x, t_y = t_preds_list[0][0], t_preds_list[0][1]

    if type(s_preds_list[0]) is int:
        sys.exit("s_preds_list[0]  should be a list")
    if type(s_preds_list[0]) is list:  # FIXME RE-order s_pres_list[0] for intersection with globl_mapped_predicates
        if s_preds_list[0] == []:
            if len(s_preds_list) > 0:
                if s_preds_list[0] == []:
                    current_options = []
                else:
                    current_options = s_preds_list[1:][0]
            else:
                current_options = []
        elif type(s_preds_list[0][0]) is list:  # alternates list
            current_options = s_preds_list[0]
        else:
            current_options = [s_preds_list[0]]  # wrap the single pred within a list
    else:
        sys.exit("HOGS.py Error - s_preds_list malformed :-(")
    candidates = []
    for singlePred in current_options:  # from s_preds_list
        t_subj, t_obj, s_subj, s_obj, topology_dist = singlePred
        #t_preds = t_grf.get_edge_data(t_subj, t_obj)  #[0]
        #s_preds = s_grf.get_edge_data(s_subj, s_obj)
        #t_vect = tgt_edge_vector_dict[t_subj, t_obj]
        #s_vect = src_edge_vector_dict[s_subj, s_obj]
        if semantics:
            t_rels = return_edges_between_these_objects(t_subj, t_obj, t_grf)
            s_rels = return_edges_between_these_objects(s_subj, s_obj, s_grf)
            # FIXME find most similar relation pair
            reslt = align_words_by_s2v(t_rels, s_rels)
            t_reln, s_reln, reln_dist = reslt[0][0], reslt[0][1], reslt[0][2]
            subj_dist = similarity(t_subj + "|NOUN", s_subj + "|NOUN")
            obj_dist = similarity(t_obj + "|NOUN", s_obj + "|NOUN")
        else:
            t_reln, s_reln = None, None
            subj_dist, reln_dist, obj_dist = 0, 0, 0

        mapped_subjects = check_if_already_mapped(t_subj, s_subj, globl_mapped_predicates)  # mapped together
        mapped_objects = check_if_already_mapped(t_obj, s_obj, globl_mapped_predicates)
        unmapped_subjects = check_if_both_unmapped(t_subj, s_subj, globl_mapped_predicates)  # unmapped target
        unmapped_objects = check_if_both_unmapped(t_obj, s_obj, globl_mapped_predicates)
        already_mapped_source = check_if_source_already_mapped([s_subj, s_reln, s_obj], globl_mapped_predicates)  # check if source already mapped to something else
        composite_distance = reln_dist + subj_dist + obj_dist + topology_dist
        if t_subj == t_obj and s_subj == s_obj:  # both reflexive relations. ok.
            pass  # that should be ok.
        elif t_subj == t_obj or s_subj == s_obj:  # ONLY one reflexive (not both, detected above) - bad. Skip
            continue
        if mapped_subjects and mapped_objects:  # both arguments co-mapped - excellent
            if check_if_already_mapped_together(t_subj, t_obj, s_subj, s_obj, globl_mapped_predicates):
                continue
            else:
                pass
        elif unmapped_subjects and unmapped_objects:
            pass  # great #TODO change to break for fast, less optimal best-first solutions?
        elif (mapped_subjects and unmapped_objects) or (unmapped_subjects and mapped_objects):
            # continue # pass # was pass, now
            pass
        elif already_mapped_source:  # unavailable for mapping 22 Feb. 24
            continue  # no match
        elif mapped_subjects:  # and unmapped_objects  # Bonus for intersecting with the current mapping
            composite_distance = composite_distance / 4.0
        elif not unmapped_subjects:  # nope
            # composite_distance = composite_distance # max_topology_distance
            continue
        elif mapped_objects:
            composite_distance = composite_distance / 2.0
        elif not unmapped_objects:
            continue
        else:
            pass  # unexpected
        # Reflexivity condition
        if s_subj == s_obj or t_subj == t_obj:    # Reflexive relations? detect and duplicate
            if (s_subj == s_obj and t_subj != t_obj) or (s_subj != s_obj and t_subj == t_obj):
                composite_distance = max_topology_distance   # composite_product = (reln_dist*4 + subj_dist + obj_dist) * topology_dist * h_prime
            else:
                candidates = candidates + [[composite_distance, s_subj, s_reln, s_obj]] # add mapping option
        else:
            candidates = candidates + [[composite_distance, s_subj, s_reln, s_obj]] # single_pred[0:5]
    candidates.sort(key=lambda x: x[0])
    # subj_in, subj_out, s_o, o_s, obj_in, obj_out, sub_pr, obj_pr, sub_sub, obj_obj, S,V,O
    for dist, s_subj, s_reln, s_obj in candidates:  # assign best
        if semantics and relational_distance(t_reln, s_reln) > max_relational_distance:
            continue
        elif conceptual_distance(t_reln, s_reln) > max_conceptual_distance:
            continue
        t_vect = tgt_edge_vector_dict[t_subj, t_obj]
        s_vect = src_edge_vector_dict[s_subj, s_obj]
        # TODO elif compare_vectors(t_vect, s_vect) > max_vector_distance:
        #    continue
        candidate_pair = [[t_subj, t_reln, t_obj], [s_subj, s_reln, s_obj], dist]
                        # add compatiable other multi-edges - cascade_mapping()
        if (check_if_already_mapped(t_subj, s_subj, globl_mapped_predicates) or    # add candidate_pair to mapping
                check_if_both_unmapped(t_subj, s_subj, globl_mapped_predicates)):
            if check_if_already_mapped(t_obj, s_obj, globl_mapped_predicates) or \
                    check_if_both_unmapped(t_obj, s_obj, globl_mapped_predicates):
                new_list = globl_mapped_predicates
                t_num_edges = num_edges_connecting(t_grf, t_subj, t_obj)
                s_num_edges = num_edges_connecting(s_grf, s_subj, s_obj)

                if t_num_edges > 1 and s_num_edges > 1:  # multi_edges
                    compatible_edges = add_consistent_multi_edges_to_mapping(t_grf, s_grf, t_subj, t_reln, t_obj,
                                            s_subj, s_reln, s_obj, candidate_pair, globl_mapped_predicates, semantics)
                    if compatible_edges: # FIXME check for duplications first
                        new_list = globl_mapped_predicates + compatible_edges # candidate pair included, as appropriate
                else:
                    new_list = globl_mapped_predicates + [candidate_pair]
                return explore_mapping_space(t_grf, s_grf, t_preds_list[1:], s_preds_list[1:], new_list, semantics)
    return explore_mapping_space(t_grf, s_grf, t_preds_list[1:], s_preds_list[1:], globl_mapped_predicates, semantics) # skip candidate_pair
    print(" **REACHED HERE**")
    #return explore_mapping_space(t_grf, s_grf, t_preds_list[1:], s_preds_list, globl_mapped_predicates, semantics)  # FIXME remove
    #return explore_mapping_space(t_grf, s_grf, t_preds_list, s_preds_list[1:], globl_mapped_predicates, semantics)  # FIXME remove


def return_edges_between_these_objects(subj, obj, thisGraph):
    """ returns a list of verbs (directed link labels) between objects - or else [] """
    res = []
    for (s, o, relation) in thisGraph.edges.data('label'):
        if (s == subj) and (o == obj):
            res.append(relation)
    return res
# returnEdgesBetweenTheseObjects('woman','bus', targetGraph)


def return_dict_from_mapping_list(pair_list):
    if pair_list == []:
        return {}
    rslt = {}
    for (t,s,val) in pair_list:
        rslt[t[0]] = s[0]
        rslt[t[2]] = s[2]
    return rslt


def num_edges_connecting(grf, subj, obj):
    n1 = grf.number_of_edges(subj, obj)
    if subj != obj:
        n2 = grf.number_of_edges(obj, subj)
    else:
        n2 = 0
    return n1 + n2

def align_words_by_s2v(lis1, lis2):
    if lis1 == [] or lis2 == []:
        return []
    elif len(lis1) >0 and len(lis2) > 0 and lis1[0] == lis2[0]:
        return [[lis1[0], lis1[0], 0]]  # lis1[0]  # possibly None
    overall_list = []
    for w1 in lis1:  # list all possible pairings -> overall_list
        tmp_lis = []
        for w2 in lis2:
            tmp_lis += [[relational_distance(w1, w2), w2]]
        tmp_lis.sort(key=lambda a: a[0])
        overall_list.append([w1, tmp_lis])
    overall_list.sort(key=lambda a: a[0][0], reverse=True)
    rslt, used_list, i = [], [], 0
    while i < len(overall_list):  # min(len(lis1), len(lis2)):
        j = 0
        while j < len(overall_list[i][1]):
            if overall_list[i][1][j][1] not in used_list:
                rslt += [[overall_list[i][0], overall_list[i][1][j][1], overall_list[i][1][j][0]]]
                used_list += [overall_list[i][1][j][1]]
                j = len(overall_list[i][1]) + 1
            else:
                j += 1
        i += 1
    return rslt
# align_words_by_s2v(['walk', 'shoot', 'pledge'], ['give', 'knew', 'saw', 'glided'])


def add_consistent_multi_edges_to_mapping(t_grf, s_grf, t_subj, t_reln, t_obj, s_subj, s_reln, s_obj,
                                          existing_pair, globl_mapped_predicates, semantics):
    """ Find the best semantic mappings between pairs of multi-edges relations. """
    compatible_pairs = []
    n1 = num_edges_connecting(t_grf, t_subj, t_obj)
    n2 = num_edges_connecting(s_grf, s_subj, s_obj)
    has_labels_flag = len(list(nx.get_edge_attributes(t_grf, "label"))) > 0
    if n1 > 1 and n2 > 1:
        t_rels_list_1 = list(t_grf.edges(t_subj, data='label', default="temp"))  # missing
        s_rels_list_1 = list(s_grf.edges(s_subj, data='label', default="temp"))
        t_rels_list_reverse = list(t_grf.in_edges(t_subj, data='label', default="temp"))
        s_rels_list_reverse = list(s_grf.in_edges(s_subj,  data='label', default="temp"))

        t_rels_1 = [rel for x, y, rel in t_rels_list_1 if x == t_subj and y == t_obj]
        s_rels_1 = [rel for x, y, rel in s_rels_list_1 if x == s_subj and y == s_obj]
        t_rels_2_rev = [rel for x,y,rel in t_rels_list_reverse if x == t_obj and y == t_subj]
        # s_rels_2_rev = [rel['label'] for x, y, rel in s_rels_list_reverse if x == s_obj and y == s_subj]
        s_rels_2_rev = [rel for x,y,rel in s_rels_list_reverse if x == s_obj and y == s_subj]
        overlap_1 = intersection(t_rels_1, s_rels_1)         # Identical words first
        overlap_2 = intersection(t_rels_2_rev, s_rels_2_rev)
        for rel in overlap_1:
            compatible_pairs += [[[t_subj, rel, t_obj], [s_subj, rel, s_obj], 0.0]]
        if t_subj != t_obj:
            for rel in overlap_2:
                compatible_pairs += [[[t_obj, rel, t_subj], [s_obj, rel, s_subj], 0.0]]
        remaining_t_rels = list_diff(t_rels_1, overlap_1)
        remaining_s_rels = list_diff(s_rels_1, overlap_1)
        pairs_1 = align_words_by_s2v(remaining_t_rels, remaining_s_rels)
        pairs_2_rev = align_words_by_s2v(list_diff(t_rels_2_rev, overlap_2), list_diff(s_rels_2_rev, overlap_2))
        for a,b, dis in pairs_1:
            #if not check_if_source_and_target_already_mapped([t_subj, a, t_obj], [s_subj, b, s_obj], globl_mapped_predicates):
            compatible_pairs += [[[t_subj, a, t_obj], [s_subj, b, s_obj], dis]]
        for a,b, dis in pairs_2_rev:
            compatible_pairs += [[[t_subj, a, t_obj], [s_subj, b, s_obj], dis]]
        if not has_labels_flag:
            new_pairs = []
            for t,s,val in compatible_pairs:
                new_pairs += [ [[t[0], None, t[2]], [s[0], None, s[2]], val] ]
            compatible_pairs = new_pairs
    return compatible_pairs


def list_diff(li1, li2):
    temp3 = []
    for element in li1:
        if element not in li2:
            temp3.append(element)
    return temp3


def check_if_source_already_mapped(s_pred, globl_mapped_predicates):
    if s_pred == [] or globl_mapped_predicates == []:
        return False
    if len(s_pred) == 3:
        for mapped in globl_mapped_predicates:
            if s_pred[0] == mapped[1][0] and s_pred[1] == mapped[1][1] and s_pred[2] == mapped[1][2]:
                return True
    return False

def check_if_source_and_target_already_mapped(t_pred, s_pred, globl_mapped_predicates):
    if globl_mapped_predicates == []:
        return True
    elif t_pred == [] or s_pred == []:
        return False
    for mapped in globl_mapped_predicates:
        if t_pred[0] == mapped[0][0] and t_pred[1] == mapped[0][1] and t_pred[2] == mapped[0][2] and \
            s_pred[0] == mapped[1][0] and s_pred[1] == mapped[1][1] and s_pred[2] == mapped[1][2]:
            return True
    return False


def check_if_already_mapped(tgt_token, src_token, globl_mapped_predicates): # 2 mapped tokens
    """ boolean, irrespective of subject or object role"""
    if globl_mapped_predicates == []:
        return False
    for x in globl_mapped_predicates:  # FIXME returns false positive :-(
        t_s, t_v, t_o = x[0]
        s_s, s_v, s_o = x[1]
        if tgt_token == t_s:
            if src_token == s_s:
                return True
        elif tgt_token == t_o:
            if src_token == s_o:
                return True
    return False
gmp = [[['department', 'in', '1987'],       ['faculty', 'in', 'department'], 2.902367303007268],
       [['members', 'of', 'department'],    ['many', 'of', 'faculty'], 3.2780674337924456],
       [['faculty', 'in', 'department'],    ['arguments', 'is', 'faculty'], 2.585683170097039],
       [['department', 'awarded', 'Award'], ['faculty', 'are', 'inaccessible'], 3.0301199055244874],
       [['z', 'awarded', 'z'], ['z', 'are', 'z'], 3.0301199055244874]]
#print(check_if_already_mapped('faculty','department',  gmp))  # False
#print(check_if_already_mapped('1987','faculty',  gmp))  # False
#print(check_if_already_mapped('z', 'z', gmp))  # True
#print(check_if_already_mapped('department', 'faculty', gmp))  # True
#print(check_if_already_mapped('1987', 'department', gmp))  # True

def check_if_already_mapped_together(t_subj, t_obj, s_subj, s_obj, globl_mapped_predicates): # 2 mapped tokens
    """ boolean, irrespective of subject or object role"""
    if globl_mapped_predicates == []:
        return False
    for x in globl_mapped_predicates:
        t_s, t_v, t_o = x[0]
        s_s, s_v, s_o = x[1]
        if t_subj == t_s and t_obj == t_o and s_subj == s_s and s_obj == s_o:
                return True
    return False


def check_if_both_unmapped(t_subj, s_subj, globl_mapped_predicates):
    """ Check if both are unmapped and are thus free to form any new mapping """
    if globl_mapped_predicates == []:
        return True
    for x in globl_mapped_predicates:
        t_s, t_v, t_o = x[0]  # target
        s_s, s_v, s_o = x[1]
        if t_subj == t_s or s_subj == s_s:
            return False
        elif t_subj == t_o or s_subj == s_o:
            return False
    return True
# print(check_if_both_unmapped('department','faculty',  gmp))  # False
# print(check_if_both_unmapped('1987','cat',  gmp))  # False
# print(check_if_both_unmapped('cat', 'faculty', gmp))  # False
# print(check_if_both_unmapped('cat', 'dog', gmp))  # True
# print(check_if_both_unmapped('x', 'y', gmp))  # True
# stop()


def evaluate_mapping(target_graph, source_graph, globl_mapped_predicates, semantics=True):
    """[[['hawk_Karla_she', 'saw', 'hunter'], ['hawk_Karla_she', 'know', 'hunter'], 0.715677797794342], [['h"""
    global mapping
    mapping = dict()
    relatio_structural_dist = 0
    paired_relations = []
    rel_s2v, con_s2v, con_count, rel_count, scor = 0, 0, 0, 0, 0
    unmapped_target_preds = return_unmapped_target_preds(target_graph, globl_mapped_predicates, semantics)
    unmapped_source_preds = return_unmapped_source_preds(source_graph, globl_mapped_predicates, semantics)
    for t_pred, s_pred, val in globl_mapped_predicates:  # full predicates
        relatio_structural_dist += val
        if t_pred == [] or s_pred == []:
            continue
        elif t_pred[0] in mapping.keys():
            if mapping[t_pred[0]] == s_pred[0]:  # extend the mapping
                scor += 0.5
            else:
                print(" Mis-Mapping 1 in DFS ")
                sys.exit(" Mis-Mapping 1 in DFS ")
        else:
            if s_pred[0] not in mapping.values():
                mapping[t_pred[0]] = s_pred[0] # new mapping
                scor += 0.25
            else:
                sys.exit("Mis Mapping 1 b")
        if t_pred[2] in mapping.keys():  # PTNT role
            if mapping[t_pred[2]] == s_pred[2]:  # extend the mapping
                scor += 0.5
            else:
                print(t_pred, "    ", s_pred)
                sys.exit(" Mis-Mapping 2 in DFS")
        else:
            if s_pred[2] not in mapping.values():
                mapping[t_pred[2]] = s_pred[2]  # new mapping
                scor += 0.25
            else:
                sys.exit(" Mis-Mapping 2 PTNT in DFS ")
        paired_relations.append([t_pred[1], s_pred[1]])
        # rel_s2v += relational_distance(t_pred[1], s_pred[1])
    extra_mapped_preds, mapping_item_pairs = \
        possible_mop_up_achievable(unmapped_target_preds, unmapped_source_preds, bestEverPredMapping, mapping)
    if extra_mapped_preds != []:
        dud = 0  # globl_mapped_predicates.append(extra_mapped_preds[0])  # globl_mapped_predicates.extend(mapping_extra)
        # TODO delete these 2 lines?
    #for x in paired_relations:
    #    print("MAP:", x)
    if semantics:
        for k,v in mapping.items():
            con_count += 1
            con_s2v += conceptual_distance(k,v)
        for t_pred, s_pred, val in globl_mapped_predicates:
            if t_pred[1] != None and s_pred[1] != None:
                rel_count += 1
                rel_s2v += relational_distance(t_pred[1], s_pred[1])
    return relatio_structural_dist, rel_s2v, rel_count, con_s2v, con_count


def return_unmapped_target_preds(target_graph, globl_mapped_predicates, semantics):
    unmapped = []
    for s,o,r in target_graph.edges(data=True):
        if semantics:
            rel = r['label']
        else:
            rel = r
        found = False
        for tgt, src, val in globl_mapped_predicates:
            if not found and s == tgt[0] and o == tgt[2] and (rel == tgt[1] or (rel == None or tgt[1] == None)):
                found = True
                break
        if found == False:
            if r:
                unmapped.append([s,r['label'],o])
            else:
                unmapped.append([s, None, o])
    return unmapped


def return_unmapped_source_preds(source_graph, globl_mapped_predicates, semantics):  # candidate inferences
    unmapped = []
    for s,o,r in source_graph.edges(data=True):
        if semantics:
            rel = r['label']
        else:
            rel = r
        found = False
        for tgt, src,_ in globl_mapped_predicates:
            if not found and s == src[0] and o == src[2] and (rel == src[1] or (rel == None or src[1] == None)):
                found = True
                break
        if found == False:
            if r:
                unmapped.append([s,r['label'],o])
            else:
                unmapped.append([s, None, o])
    return unmapped


def possible_mop_up_achievable(t_preds_list, s_preds_list, bestEverPredMapping, mapped_concepts, try_harder=False):
    """ Opportunistic extension to mapping, caused by narrowness of beam search.
    Compare Unmapped Target Predicates with Unmapped Source Predicates."""
    if t_preds_list == [] or s_preds_list == []:
        return [], mapped_concepts
    mapping_extra, mx2 = [], []
    if try_harder == True:
        dud = 0
    for t_subj, t_rel, t_obj in t_preds_list:
        for s_subj, s_rel, s_obj in s_preds_list:
            if check_if_already_mapped(t_subj, s_subj, bestEverPredMapping) and \
                    relational_distance(t_rel, s_rel) < max_relational_distance and \
                    check_if_both_unmapped(t_obj, s_obj, bestEverPredMapping):
                mapping_extra.append([[t_subj, t_rel, t_obj], [s_subj, s_rel, s_obj], 1.1])
                bestEverPredMapping.append([[t_subj, t_rel, t_obj], [s_subj, s_rel, s_obj], 1.1])  # Aug 23
                mapped_concepts[t_obj] = s_obj
            elif check_if_already_mapped(t_obj, s_obj, bestEverPredMapping) and \
                    relational_distance(t_rel, s_rel) < max_relational_distance and \
                    check_if_both_unmapped(t_subj, s_subj, bestEverPredMapping):
                mapping_extra += [[[t_subj, t_rel, t_obj], [s_subj, s_rel, s_obj], 1.11]]
                bestEverPredMapping.append([[t_subj, t_rel, t_obj], [s_subj, s_rel, s_obj], 1.11])  # Aug 23
                mapped_concepts[t_subj] = s_subj
            elif check_if_already_mapped(t_subj, s_subj, bestEverPredMapping) and \
                    check_if_already_mapped(t_obj, s_obj, bestEverPredMapping) and \
                    relational_distance(t_rel, s_rel) < max_relational_distance:
                #print("EXTRA compatible ", end=" ")  # TODO: explore in depth
                mapping_extra += [[[t_subj, t_rel, t_obj], [s_subj, s_rel, s_obj], 1.11]]
                bestEverPredMapping.append([[t_subj, t_rel, t_obj], [s_subj, s_rel, s_obj], 1.11])
            elif try_harder == True and check_if_both_unmapped(t_subj, s_subj, bestEverPredMapping) \
                    and check_if_both_unmapped(t_obj, s_obj, bestEverPredMapping) and \
                    relational_distance(t_rel, s_rel) < max_relational_distance:
                mapping_extra += [[[t_subj, t_rel, t_obj], [s_subj, s_rel, s_obj], 1.34]]
                bestEverPredMapping.append([[t_subj, t_rel, t_obj], [s_subj, s_rel, s_obj], 1.34])  # Sept 23
                mapped_concepts[t_subj] = s_subj
    if mapping_extra != []:
        for tgt, src, val in mapping_extra:
            if tgt in t_preds_list and src in s_preds_list:
                t_preds_list.remove(tgt)
                s_preds_list.remove(src)
            else:
                print("  ALARM BELLS!!! ", end="")
        #print(" XXXXXXX Map Extension by ", len(mapping_extra), end="  ") # TODO: explore in depth
    if try_harder == False:
        mx2, mp2 = possible_mop_up_achievable(t_preds_list, s_preds_list,
                                            bestEverPredMapping, mapped_concepts, try_harder = True)
    if mx2 != []:
        # print("EXTRA hard found ", len(mx2), end = " XX    ")  # TODO Explore in depth
        mapping_extra = mapping_extra + mx2
    return mapping_extra, mapped_concepts  # return bestEverPredMapping, mapped_concepts


def relational_distance(t_in, s_in):  # using s2v sense2vec
    global term_separator
    global s2v_verb_cache, wnl
    if t_in == s_in:  # contain == contains; contains =/= parameter
        return 0
    elif mode == "Code":  # identical relations only, identicality constraint
        return 1
    # print("REL-Dis:", t_in, s_in, end=" ")
    if isinstance(t_in, dict):
        dud=0
    t_reln = t_in.split(term_separator)[0]
    s_reln = s_in.split(term_separator)[0]
    if t_in == "" or s_in == "" or t_reln == "" or s_reln == "":  # treat this as an error?
        return 0.9797
    if isinstance(t_reln, dict) and isinstance(s_reln, dict) and t_reln[0] == s_reln[0]:
        return 0.001
    elif second_head(t_reln) == second_head(s_reln):
        return 0.01
    elif t_reln + "-" + s_reln in s2v_verb_cache:
        sim_score = s2v_verb_cache[t_reln + "-" + s_reln]
        return sim_score
    elif s_reln + "-" + t_reln in s2v_verb_cache:
        sim_score = s2v_verb_cache[s_reln + "-" + t_reln]
        return sim_score
    else:
        if s2v.get_freq(t_reln + '|VERB') is None or s2v.get_freq(s_reln + '|VERB') is None:
            print("H2-WNL: ", t_reln, "...", s_reln, end=" ")
            t_root = wnl.lemmatize(t_reln)
            s_root = wnl.lemmatize(s_reln)
            if s2v.get_freq(t_root + '|VERB') is None or s2v.get_freq(s_root + '|VERB') is None:
                return 1.0
            else:
                sim_score = 1 - s2v.similarity([t_root + '|VERB'], [s_root + '|VERB'])
                s2v_verb_cache[t_reln + "-" + s_reln] = sim_score
                return sim_score
        else:
            sim_score = s2v.similarity([t_reln + '|VERB'], [s_reln + '|VERB'])
            s2v_verb_cache[t_reln + "-" + s_reln] = sim_score
            return 1 - sim_score


def conceptual_distance(str1, str2):
    """for simple conceptual similarity"""
    global term_separator
    global s2v_noun_cache
    if str1 == str2:
        return 0.00000
    if isinstance(str1, (int, float)):  # numeric generated graphs
        return abs(str1 - str2)
    arg1 = str1.replace(".", ":").split(term_separator)  # either : or _
    arg2 = str2.replace(".", ":").split(term_separator)
    if mode == "Code":  # check for overlap '1. Block:1' and '158. Block:158'
        if arg1 == arg2:
            return 0
        if arg1[0] == arg2[0]:
            if arg1[1] == arg2[1]:
                return 0.0
            else:
                return 0.05
        if len(arg1) > 1 and len(arg2) > 1:
            if arg1[1] == arg2[1]:
                return 0.02
            else:
                return 1
        else:
            return 1
        return 1  # catch-all
        #elif arg1[1] == arg2[1]:
        #    inter_sectn = list(set(arg1) & set(arg2))
        #    if len(inter_sectn) > 0:
        #        return min(0.1, 0.5 / len(inter_sectn))
        #    else:
        #        return 0.1
        #else:
        #    return 0.99
    elif mode == "English":
        if arg1[0] + "-" + arg2[0] in s2v_noun_cache:
            sim_score = s2v_noun_cache[arg1[0] + "-" + arg2[0]]
            return sim_score
        elif arg2[0] + "-" + arg1[0] in s2v_noun_cache:
            sim_score = s2v_noun_cache[arg2[0] + "-" + arg1[0]]
            return sim_score
        sim_1 = s2v.get_freq(arg1[0] + '|NOUN')
        sim_2 = s2v.get_freq(arg2[0] + '|NOUN')
        if not sim_1 and not sim_2:
            return 0.7  # both novel words/nouns
        elif not sim_1 or not sim_2:
            return 1.0
        else:
            sim_score = s2v.similarity([arg1[0] + '|NOUN'], [arg2[0] + '|NOUN'])
            s2v_noun_cache[arg1[0] + "-" + arg2[0]] = sim_score
            return 1 - sim_score
    else:
        return 1.5
#print(conceptual_distance("dog", "puppy"))
#print(conceptual_distance("dog", "chair"))
# print(conceptual_distance('1. Block:1', '158. Block:158'))
# stop()

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def return_sorted_predicates(grf, uniq_grf_pairs):
    """ Sort input predicates according to their combined arity.
     subj_in, subj_out, s_o, o_s, obj_in, obj_out, sub_pr, obj_pr, sub_sub, obj_obj, S,V,O  """
    edge_vector_dict = {}
    pred_list = []
    pr = nx.pagerank(grf, alpha=0.85)
    grf_rev = generate_reversed_edge_graph(grf)
    pr_rev = nx.pagerank(grf_rev, alpha=0.85)
    for s, o in uniq_grf_pairs:  # uniq_grf_pairs FIXME: wrong collection listed here, replace
        if type(grf).__name__ == "MultiDiGraph" or type(grf).__name__ == "DiGraph":
            into_s = grf.in_degree(s)
            out_from_s = grf.out_degree(s)
            into_o = grf.in_degree(o)
            out_from_o = grf.out_degree(o)
        else:  # Graph() and MultiGraph()
            into_s = grf.degree(s)
            out_from_s = grf.degree(s)
            into_o = grf.degree(o)
            out_from_o = grf.degree(o)
        s_to_o = grf.number_of_edges(s, o)
        o_to_s = grf.number_of_edges(o, s)
        s_to_s = grf.number_of_edges(s, s)
        o_to_o = grf.number_of_edges(o, o)
        w = pr[s] ** (1./3.) / 4.  # RageRank values descriptors
        x = pr[o] ** (1./3.) / 4.
        w1 = pr_rev[s] ** (1./3.) / 4.
        x1 = pr_rev[o] ** (1./3.) / 4.
        lis = [into_s, out_from_s, s_to_o, o_to_s, into_o, out_from_o, w, x, w1, x1, s_to_s, o_to_o]
        tot = sum(lis)
        pred_list.append([tot, s, o])
        edge_vector_dict[s, o] = np.array(lis)
    z = sorted(pred_list, key=lambda x: x[0], reverse=True)
    #z2 = sorted(pred_list, key=sum, reverse=True)
    return z, edge_vector_dict


def my_sub_sort(lst):
    digits = sum(lst[0:12])  # FIXME sum an arbitrary length list
    res2 = sum(lst)
    return digits


def generate_reversed_edge_graph(g1):
    g2 = nx.MultiDiGraph()
    for u,v,a in g1.edges(data=True):
        g2.add_edge(v,u, label=a)
    return g2


# ##############################################################################################################
# ##############################################################################################################
# ##############################################################################################################

# You can generate m lists [1...n] and use itertools.product to get the cartesian product between these lists.
from itertools import product

def verify(G, H, f):
   homomorphism = True
   for edge in G:
       if not ((f[edge[0]], f[edge[1]]) in H):
           homomorphism = False
           break
   return homomorphism


def solve(G, H, n, m):
   rangeG = [i for i in range(n)]
   assignments = list(product(rangeG, repeat=m))
   cnt = 0
   for f in assignments:
       if verify(G, H, f):
           cnt += 1
   return cnt

# G = {(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0)}
# H = {(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)}
# print(" solve(G, H, 4, 4) -> ", solve(G, H, 4, 4), end=" ")
# H2 = {(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0)}
# print(" solve(G, H2, 4, 4) -> ", solve(G, H, 4, 4), end=" ")
# stop()
# ##############################################################################################################
# ##############################################################################################################
# ##############################################################################################################


def build_graph_from_triple_list(triple_list):  # unused?
    temp_graph2 = nx.MultiDiGraph()
    temp_graph2.clear()
    temp_graph2.graph['Graphid'] = "Dud Name"
    previous_subj = last_v = previous_obj = ""
    for triple in triple_list:
        if len(triple) == 3:  # subject, verb, obj
            noun1, verb, noun2 = triple
        elif len(triple) == 4:  # methodName, subject, verb, obj
            methodName, noun1, verb, noun2 = triple
        elif len(triple) == 0:
            continue
        elif mode == 'code' or len(triple) == 6:  # Code Graphs
            if triple[0] == "CodeContracts":  # skip the contracts?
                pass  # break
            if len(triple) == 6:
                methodName, noun1, verb, noun2, nr1, nr2 = triple
                noun1 = noun1.strip() + term_separator + nr1
                noun2 = noun2.strip() + term_separator + nr2
            elif len(triple) == 3:
                exit("BGfC len  triple ==  3 error")
                noun1, verb, noun2 = triple
            elif len(triple) != 3:
                print("Possibly embedded SQL in: ")
                pass
            else:
                noun1, verb, noun2 = triple
        if isinstance(noun1, str):
            noun1 = noun1.strip()  # remove spaces
            verb = verb.strip()
            noun2 = noun2.strip()
        if mode == 'English':
            #if skip_prepositions and preposition_test(verb):
            #    continue
            if (noun1 == previous_subj) and (noun2 == previous_obj):  # and preposition_test(verb):
                verb = last_v + "_" + verb
                print(verb, end=" ")
                temp_graph2.remove_edge(noun1, noun2)
            elif noun1 == "NOUN":  # skip header data from files
                continue

            if isinstance(noun1, str):
                if len(noun1.split(term_separator)) > 1:
                    noun1 = parse_new_coref_chain(noun1)
                if len(noun2.split(term_separator)) > 1:
                    noun2 = parse_new_coref_chain(noun2)

        temp_graph2.add_node(noun1, label=noun1)
        temp_graph2.add_node(noun2, label=noun2)
        temp_graph2.add_edge(noun1, noun2, label=verb)

        if mode == 'English':
            previous_subj = noun1
            last_v = verb  # phrasal verbs; read_over, read_up, read_out
            previous_obj = noun2
    returnGraph = nx.MultiDiGraph(temp_graph2)  # no need for a .copy()
    return returnGraph  # results in the canonical  version of the graph :-)



def addRelationsToMapping(target_graph, source_graph, mapping_dict, s_decoding, t_decoding):  # LCS_Number
    "new one. GM.mapping={(t,s), (t2,s2)...}"
    this_mapping = mapping_dict.copy()  # was GM.mapping.
    mappedFullPredicates = []
    rel_sim_total = 0
    rel_sim_count = 0
    all_target_edges = returnEdgesAsList(target_graph)
    source_edge_list = returnEdgesAsList(source_graph)
    for tNoun1, tRelation, tNoun2 in all_target_edges:
        tNoun1_num = t_decoding[tNoun1]  # decoding ={label, number }
        tNoun2_num = t_decoding[tNoun2]
        if tNoun1_num in this_mapping.keys() and tNoun2_num in this_mapping.keys():  # and \
            #    this_mapping[tNoun1_num] in this_mapping.values() and this_mapping[tNoun2_num] in this_mapping.values():
            sNoun1_num = this_mapping[tNoun1_num]
            sNoun2_num = this_mapping[tNoun2_num]
            if not (sNoun1_num in s_decoding.keys() and sNoun2_num in s_decoding.keys()):
                break
            sNoun1 = s_decoding[sNoun1_num]  # what if null-  test require
            sNoun2 = s_decoding[sNoun2_num]
            source_verbs = returnEdgesBetweenTheseObjects_predList(sNoun1, sNoun2, source_edge_list)
            closest_sim = 0.0
            nearest_s_verb = "nULl"
            simmy = 0
            for s_verb in source_verbs:
                if s_verb:
                    if tRelation == s_verb:
                        simmy = 1
                    elif s2v.get_freq(head_word(tRelation) + '|VERB') is not None and \
                            s2v.get_freq(head_word(s_verb) + '|VERB') is not None:
                        simmy = s2v.similarity([head_word(tRelation) + '|VERB'], [head_word(s_verb) + '|VERB'])
                    #else:
                    #    print("**~ S2V", tRelation, s_verb, end="   ")
                if simmy >= closest_sim:
                    closest_sim = simmy
                    nearest_s_verb = s_verb
                    cached_predicate_mapping = [[tNoun1, tRelation, tNoun2, sNoun1, nearest_s_verb, sNoun2]]
            if not nearest_s_verb == 'nULl':
                rel_sim_total += closest_sim
                rel_sim_count += 1
                this_mapping[tRelation] = nearest_s_verb
                mappedFullPredicates += cached_predicate_mapping
                source_edge_list.remove([sNoun1, nearest_s_verb, sNoun2])
    mappedFullPredicates.insert(0, rel_sim_total)
    return mappedFullPredicates  # rel_sim, rel_sim_count,


def return_list_of_mapped_concepts(list_of_mapped_preds):
    """ Return a dictionary of paired concepts/nouns. No tests conducted. """
    mapping_dict = {}
    for t_pred, s_pred, val in list_of_mapped_preds:
        mapping_dict[t_pred[0]] = s_pred[0]
        mapping_dict[t_pred[2]] = s_pred[2]
    res = list(mapping_dict.items())
    return res

# align_words_by_s2v(['walk', 'shoot', 'pledge'], ['give', 'knew', 'saw', 'glided'])
# stop()

def project_implicit_word_pairs_BACKUP():
    import os
    global mode
    # mode = "English"
    base_path = "C:/Users/dodonoghue/Documents/Python-Me/data/Bias - implicit/"
    source_files = os.listdir(base_path)
    these_files = [i for i in source_files if i.endswith('.txt')]
    for x in these_files:
        with open(base_path + x, 'r') as in_file:
            print("FILE", x)
            text_contents = ""
            for row in in_file:
                if len(row) > 1:
                    text_contents = row.split(",")
                    w0, w1 = text_contents[0], text_contents[1].strip()
                    if s2v.get_freq(w0 + '|NOUN') is None or s2v.get_freq(w1 + '|NOUN') is None:
                        print("na")
                    else:
                        print(s2v.similarity(w0 + '|NOUN', w1 + '|NOUN'))
                    #print(w0, w1, conceptual_distance(w0, w1))  #Sense2Vec
                    if w0 == "career":
                        dud = 0
                    #doc1 = nlp(text_contents[0])
                    #doc2 = nlp(text_contents[1])
                    #print(doc1.similarity(doc2))
            print(" ")
            print(" ")
            print()
# project_implicit_word_pairs_BACKUP()