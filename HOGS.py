# HOGS is a heuristic algorithm that presumes that edges with greater arity
# should be given a greater priority than edges with a smaller arity.
# It explores the edge-space of graph-subgraph near isomorphism.
# loosely inspired by Nawaz, Enscore and Ham (NEH) algorithm
# local optimisation, near the global optimum.
# Edges are described by a 4-tuple of in/out degrees from a di-graph. 2 edges are compared by Wasserstein metric.
# I believe it's an admissible heuristic! A narrow search space is explored heuristically.
#
# Homomorphic Erdos-Renyi Graph Search
# Subgraph Subgraph Isomorphism Erdos-Renyi graph Search (HOGS - SSIS)
# Can I use a simple if statement to skip over the second and subsequent edges on the target graph during search?
# Homomorphic Graph Degree guided Search HGDGS
# edge_align()

import sys
import math
import networkx as nx
import random
import ShowGraphs
# import tqdm


from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

global mode
global term_separator
global max_topology_distance
global max_relational_distance, max_conceptual_distance
global numeric_offset
global global_mapped_predicates


max_topology_distance = 20  # in terms of a node's in/out degree
numeric_offset = 1000
mode = "English"
mode = 'Code'
if mode == "English":
    term_separator = "_"  # Map2Graphs.term_separator
    max_relational_distance = 1.1  # 0.9
    max_conceptual_distance = 1.1  # 0.9
elif mode == "Code":
    term_separator = ":"
    max_relational_distance = 0.5
    max_conceptual_distance = 0.0

global s2v

if False:
    from sense2vec import Sense2Vec
    # s2v = Sense2Vec().from_disk("C:/Users/user/Documents/Python-Me/Sense2Vec/s2v_reddit_2019_lg/")
    s2v = Sense2Vec().from_disk("C:/Users/dodonoghue/Documents/Python-Me/Sense2Vec/s2v_reddit_2019_lg/")
    # assert query in s2v
    #print("s2v cat dog = ", s2v.similarity(['cat' + '|NOUN'], ['dog' + '|NOUN']))


global s2v_verb_cache
s2v_verb_cache = dict()
s2v_verb_cache = {'abc': 0}  # to avoid a null key problem
global s2v_noun_cache
s2v_noun_cache = dict()
s2v_noun_cache = {'abc': 0}  # to avoid a null key problem


def find_nearest(vec):
    for key, vec in s2v.items():
        print(key, vec)


beam_size = 3  # beam breadth for beam search
epsilon = 100
current_best_mapping = []
bestEverMapping = []


def MultiDiGraphMatcher(target_graph, souce_graph):
    generate_and_explore_mapping_space(target_graph, souce_graph, False)


def get_freq(word):
    return s2v.get_freq(word)


# @staticmethod
def generate_and_explore_mapping_space(target_graph, source_graph, semantics=True):  # new one
    global current_best_mapping, bestEverMapping, globl_mapped_predicates
    global beam_size, epsilon
    current_best_mapping, bestEverMapping = [], []
    if target_graph.number_of_edges() == 0 or source_graph.number_of_edges() == 0:
        return [], 0, {}
    target_preds_with_multi = return_sorted_predicates(target_graph)  # search only over reduced space with
    target_preds = trim_multi_edges(target_preds_with_multi)     # separate iteration over self edges
    source_preds_with_multi = return_sorted_predicates(source_graph)
    source_preds = trim_multi_edges(source_preds_with_multi)
    candidate_sources = []
    for t_subj_in, t_subj_out, a_to_b, b_to_a, t_obj_in, t_obj_out, w,x,y,z, t_subj, t_reln, t_obj in target_preds:
        best_distance, composite_distance, best_subj, best_reln, best_obj \
                          = sys.maxsize, sys.maxsize, "nil", "nil", "nil"
        alternate_candidates, alternates_confirmed = [], []
        number_target_edges = target_graph.number_of_edges(t_subj, t_obj)
        for s_subj_in, s_subj_out, p_to_q, q_to_p, s_obj_in, s_obj_out, w1, x1, y1, z1, s_subj, s_reln, s_obj in source_preds:
            topology_dist = math.sqrt( euclidean_distance(
                t_subj_in, t_subj_out, a_to_b, b_to_a, t_obj_in, t_obj_out, w,  x,  y,  z,
                s_subj_in, s_subj_out, p_to_q, q_to_p, s_obj_in, s_obj_out, w1, x1, y1, z1))
            if topology_dist > max_topology_distance:
                continue
            elif (t_subj == t_obj) and (s_subj != s_obj):  # if one is a self-map && the other not ...
                continue
            elif (t_subj != t_obj) and (s_subj == s_obj):  # if one is a self-map && the other not ...
                continue
            number_source_edges = source_graph.number_of_edges(s_subj, s_obj)
            if semantics:
                if number_target_edges > 1 or number_source_edges > 1:
                    t_rels = return_relations_between_these_objects(target_graph, t_subj, t_obj)
                    s_rels = return_relations_between_these_objects(source_graph, s_subj, s_obj)
                    reln_dist, t_rel, s_rel = return_closest_relation_pair(t_rels, s_rels)  ######## BEST COMBO ###########
                    dud = 0
                else:
                    reln_dist = relational_distance(t_reln, s_reln)
                if reln_dist > max_relational_distance:
                    continue
                subj_dist = conceptual_distance(t_subj, s_subj)
                obj_dist = conceptual_distance(t_obj, s_obj)
                #if subj_dist <= max_conceptual_distance or obj_dist <= max_conceptual_distance:
                #    continue
            else:
                reln_dist, subj_dist, obj_dist = 1, 1, 1
            combo_dist = scoot_ahead(t_subj, s_subj, t_reln, s_reln, t_obj, s_obj, source_graph, target_graph, semantics)
            h_prime = combo_dist
            composite_distance = (reln_dist*15 + subj_dist + obj_dist) + topology_dist/2 + h_prime/2
            # composite_distance = (reln_dist*2 + subj_dist + obj_dist) + topology_dist + h_prime
            if composite_distance < best_distance:         # minimize distance
                best_distance = composite_distance
            alternate_candidates.append([s_subj_in, s_subj_out, p_to_q, q_to_p, s_obj_in, s_obj_out, s_subj, s_reln, s_obj,
                                         composite_distance, reln_dist, subj_dist, obj_dist, topology_dist, h_prime])
        alternate_candidates.sort(key=lambda x: x[9])  # sort by composite distance
        if len(alternate_candidates) > 0:
            alternates_confirmed = []
            for x in alternate_candidates:
                if abs(x[9] - best_distance) < epsilon: # and best_distance < 250.00
                    alternates_confirmed.append(x)  # flat list of sublists
        alternates_confirmed = alternates_confirmed[:beam_size]  # consider BEST options only, Threshold?
        # alternates_confirmed = alternates_confirmed  # consider all options
        candidate_sources.append(alternates_confirmed)  # ...(alternates_confirmed[0])
        # FIXME is it generating the correct search space for the p and r domains?
    # print(target_preds, "\n\n", candidate_sources)
    reslt = explore_mapping_space(target_graph, source_graph, target_preds, candidate_sources, [], semantics)
    zz = evaluate_mapping(target_graph, source_graph, reslt, semantics)  # bestEverMapping
    return bestEverMapping, len(bestEverMapping), mapping


def return_closest_relation_pair(t_rels, s_rels):
    if t_rels == [] or s_rels == []:
        return "nil", 1.0
    intersect_n = intersection(t_rels, s_rels)
    if intersect_n:
        return 0, intersect_n[0], intersect_n[0]
    else:
        t_rel_best, s_rel_best, rel_dist_best = "nil", "nil", 999
        for r1 in t_rels:
            for r2 in s_rels:
                rel_dist_this = relational_distance(r1, r2)
                if rel_dist_this < rel_dist_best:
                    t_rel_best, s_rel_best = r1, r2
                    rel_dist_best = rel_dist_this
        reln_dist = rel_dist_best
    return reln_dist, t_rel_best, s_rel_best


def return_relations_between_these_objects(thisGraph, subj, obj):  # from Map2Graphs
    """ returns a list of verbs (directed link labels) between objects - or else [] """
    res = []
    for (s, o, relation) in thisGraph.edges.data('label'):
        if (s == subj) and (o == obj):
            res.append(relation)
    return res


def trim_multi_edges(pred_list):  # new version
    """ [[1, 3, 3, 0, 3, 1, 'hunter_he', 'pledged', 'hawk'] ..."""
    visited_pairs = []
    reduced_edges = []
    for a, b, c, d, e, f, p,q,r,s, x, rel, y in pred_list:  # use wordnet lemmas count()
        if [x, y] in visited_pairs or [y, x] in visited_pairs:
            pass
        else:
            reduced_edges.append([a, b, c, d, e, f, p,q,r,s, x, rel, y])
            visited_pairs.append([x,y])
    return reduced_edges

######################################### Scoot Ahead ###########################################################


def return_best_in_combo(targetGraph, sourceGraph, tgt_subj, src_subj, tgt_preds, src_preds, semantics=True):
    """ Guide search by relatio-topological similarity.
        {'lake': {0: {'label': 'fed_by'}}, 'they': {0: {'label': 'conveyed'}}}) """
    result_list, min_rel_dist = [], 999
    in_t_rel_list, in_s_rel_list = [], []
    for in_tgt_nbr, reln in tgt_preds:  # find MOST similar S & T pair
        # FIXME: deal with >=2 edges between nodes. Maybe unnecessary though
        tgt_in_deg = targetGraph.in_degree(in_tgt_nbr)
        tgt_out_deg = targetGraph.out_degree(in_tgt_nbr)
        if semantics:
            temp = list(reln)[0]
            if isinstance(temp, int):  # irregularity arising from NetworkX merging nodes
                zz = reln[0]['label']
                in_t_rel_list.append([in_tgt_nbr, tgt_in_deg, tgt_out_deg, zz])
            else:
                in_t_rel_list.append([in_tgt_nbr, tgt_in_deg, tgt_out_deg, temp])
    for in_src_nbr, foovalue2 in src_preds:
        s_in_deg = sourceGraph.in_degree[in_src_nbr]
        s_out_deg = sourceGraph.out_degree[in_src_nbr]
        if semantics:
            temp = list(foovalue2)[0]
            if isinstance(temp, int):  # irregularity arising from NetworkX merging nodes
                zz = foovalue2[0]['label']
                in_s_rel_list.append([in_src_nbr, s_in_deg, s_out_deg, zz])
            else:
                in_s_rel_list.append([in_src_nbr, s_in_deg, s_out_deg, temp])

    reslt = align_edges_single_arity_only(in_t_rel_list, in_s_rel_list, semantics)
    if reslt == []:
        scr = 0
    else:
        scr = reslt[0][0]
    return reslt, scr


def return_best_out_combo(targetGraph, sourceGraph, tgt_obj, src_obj, t_preds, s_preds, semantics=True):
    """{'lake': {0: {'label': 'fed_by'}}, 'they': {0: {'label': 'conveyed'}}}) """
    result_list, min_rel_dist = [], 999
    out_t_rel_list, out_s_rel_list = [], []
    for out_t_nbr, foovalue in t_preds:
        t_in_deg = targetGraph.in_degree(out_t_nbr)
        t_out_deg = targetGraph.out_degree(out_t_nbr)
        if semantics:
            dud = list(foovalue)[0]
            if isinstance(dud, int):  # Required because of irregularity arising from merging nodes
                zz = foovalue[0]['label']  # XXX
                out_t_rel_list.append([out_t_nbr, t_in_deg, t_out_deg, zz])
            else:
                out_t_rel_list.append([out_t_nbr, t_in_deg, t_out_deg, dud])
    for out_s_nbr, foovalue2 in s_preds:
        s_in_deg = sourceGraph.in_degree(out_s_nbr)
        s_out_deg = sourceGraph.out_degree(out_s_nbr)
        if semantics:
            dud = list(foovalue2)[0]
            if isinstance(dud, int):  # Required because of irregularity arising from merging nodes NetworkX
                zz = foovalue2[0]['label']  # XXX
                out_s_rel_list.append([out_s_nbr, s_in_deg, s_out_deg, zz])
            else:
                out_s_rel_list.append([out_s_nbr, s_in_deg, s_out_deg, dud])

    reslt = align_edges_single_arity_only(out_t_rel_list, out_s_rel_list, semantics)   # Now
    if reslt == []:
        scr = 0
    else:
        scr = reslt[0][0]
    return reslt, scr


def align_edges_single_arity_only(t_rels_list, s_rels_list, semantics):
    """ Used by scoot_ahead to estimate maximal mapping between incoming links to an edge's Agnt role.
        Returns only a single best solution - best trg -> src relation mapping."""
    global max_relational_distance
    result_list, rel_dist = [], 1
    for nn1, a, b, rel1 in t_rels_list:
        for nn2, p, q, rel2 in s_rels_list:
            if semantics:
                rel_dist = max(relational_distance(rel1, rel2), 0.001)
                if rel_dist < max_relational_distance:
                    rel_dist = 1.0  # FIXME set to 1000 say?
            topo_dist = max((abs(a - p) + abs (b - q)), 0.001)
            prod_dist = rel_dist * topo_dist
            #if prod_dist < min_rel_dist:
            #    min_rel_dist = prod_dist
            result_list.append([prod_dist, rel1, rel2])
    rslt = sorted(result_list, key=lambda x:x[0])
    return rslt


def scoot_ahead(t_subj, s_subj, t_reln, s_reln, t_obj, s_obj, sourceGraph, targetGraph, semantics): #=True):
    best_in_links, in_rel_sim = return_best_in_combo(targetGraph, sourceGraph, t_subj, s_subj,
                                       targetGraph.pred[t_subj].items(), sourceGraph.pred[s_subj].items(), semantics)
    best_out_links, out_rel_sim = return_best_out_combo(targetGraph, sourceGraph, t_obj, s_obj,
                                        targetGraph.succ[t_obj].items(), sourceGraph.succ[s_obj].items(), semantics)
    combined_distance = in_rel_sim + out_rel_sim # reln_dist + in_rel_sim
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
# ############################################## EXPLORE SPACE #######################################################
# #####################################################################################################################


def explore_mapping_space(t_grf, s_grf, t_preds_list, s_preds_list, globl_mapped_predicates, semantics=True):
    """ Map the next target pred, by finding a mapping from the sources"""
    global max_topology_distance
    global bestEverMapping
    if len(t_preds_list) + len(globl_mapped_predicates) < len(bestEverMapping):  # abandon early
        return globl_mapped_predicates
    if len(globl_mapped_predicates) > len(bestEverMapping):  # compare scores, not lengths?
        #if True and len(bestEverMapping) > 0:
        #    print("¬", len(bestEverMapping), end=" ")
        bestEverMapping = globl_mapped_predicates
    if t_preds_list == [] or s_preds_list == []:
        return globl_mapped_predicates
    elif s_preds_list[0] == []:
        return explore_mapping_space(t_grf, s_grf, t_preds_list, s_preds_list[1:], globl_mapped_predicates)
    elif t_preds_list[0] == []:
        explore_mapping_space(t_grf, s_grf, t_preds_list[1:], s_preds_list, globl_mapped_predicates)
    elif t_preds_list != [] and s_preds_list[0] == []:
        explore_mapping_space(t_grf, s_grf, t_preds_list, s_preds_list[1:], globl_mapped_predicates)
        # return globl_mapped_predicates
    elif t_preds_list == [] or s_preds_list == []:
        if len(globl_mapped_predicates) > len(bestEverMapping):
            bestEverMapping = globl_mapped_predicates
        return globl_mapped_predicates

    t_subj_in, t_subj_out, a_to_b, b_to_a, t_obj_in, t_obj_out, p,q,r,s, t_subj, t_reln, t_obj = t_preds_list[0]

    if type(s_preds_list[0]) is int:  # Error
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
        sys.exit("DFS.py Error - s_preds_list malformed :-(")
    candidates = []
    for singlePred in current_options:  # from s_preds_list
        s_subj_in, s_subj_out, a_to_b, b_to_a, s_obj_in, s_obj_out, s_subj, s_reln, s_obj, composite_distance,\
            reln_dist, subj_dist, obj_dist, topology_dist, h_prime = singlePred
        dud_t_pred = t_subj, t_reln, t_obj
        dud_s_pred = s_subj, s_reln, s_obj
        mapped_subjects = check_if_already_mapped(t_subj, s_subj, globl_mapped_predicates)  # mapped together
        mapped_objects = check_if_already_mapped(t_obj, s_obj, globl_mapped_predicates)
        unmapped_subjects = check_if_both_unmapped(t_subj, s_subj, globl_mapped_predicates)  # unmapped target
        unmapped_objects = check_if_both_unmapped(t_obj, s_obj, globl_mapped_predicates)
        already_mapped_source = check_if_source_already_mapped([s_subj, s_reln, s_obj], globl_mapped_predicates) # check if source already mapped to something else
        composite_distance = reln_dist + subj_dist + obj_dist + topology_dist # +h_prime
        if t_subj == t_obj and s_subj == s_obj:  # both reflexive relations. ok.
            pass  # that should be ok.
        elif t_subj == t_obj and  s_subj != s_obj:  # ONLY one reflexive (not both, detected above) - bad. Skip
            continue
        elif t_subj != t_obj and  s_subj == s_obj:  # ONLY one reflexive (not both, detected above) - bad. Skip
            continue
        if mapped_subjects and mapped_objects:  # both arguments co-mapped - excellent
            pass
        elif unmapped_subjects and unmapped_objects:
            pass  # great
        elif (mapped_subjects and unmapped_objects) or (unmapped_subjects and mapped_objects):
            pass
        elif already_mapped_source:
            continue  # no match
        elif mapped_subjects:  # and unmapped_objects  # Bonus for intersecting with the current mapping
            composite_distance = composite_distance / 3.0
        elif not unmapped_subjects:  # nope
            continue            # composite_distance = composite_distance # max_topology_distance
        elif mapped_objects:
            composite_distance = composite_distance / 3.0
        elif not unmapped_objects:
            continue
        else:
            pass  # unexpected
        # Confirm that candidate
        if s_subj == s_obj or t_subj == t_obj:    # Reflexive relations? detect and duplicate
            if (s_subj == s_obj and t_subj != t_obj) or (s_subj != s_obj and t_subj == t_obj):
                composite_distance = max_topology_distance
            else:
                candidates = candidates + [[composite_distance, s_subj, s_reln, s_obj]]
        else:
            candidates = candidates + [[composite_distance, s_subj, s_reln, s_obj]]
        dud = 0
    candidates.sort(key=lambda x: x[0])
    # ########################## subj_in, subj_out, s_o, o_s, obj_in, obj_out, sub_pr, obj_pr, sub_sub, obj_obj, S,V,O
    for dist, s_subj, s_reln, s_obj in candidates:  # best first exploration
        if semantics and relational_distance(t_reln, s_reln) > max_relational_distance:
            continue
        candidate_pair = [[t_subj, t_reln, t_obj], [s_subj, s_reln, s_obj], dist]
        if (check_if_already_mapped(t_subj, s_subj, globl_mapped_predicates) or    # add candidate_pair to mapping
                check_if_both_unmapped(t_subj, s_subj, globl_mapped_predicates)):
            if check_if_already_mapped(t_obj, s_obj, globl_mapped_predicates) or \
                    check_if_both_unmapped(t_obj, s_obj, globl_mapped_predicates):
                new_list = globl_mapped_predicates
                t_rels = return_relations_between_these_objects(t_grf, t_subj, t_obj)
                s_rels = return_relations_between_these_objects(s_grf, s_subj, s_obj)
                t_num_edges = len(t_rels)
                s_num_edges = len(s_rels)
                if t_num_edges > 1 or s_num_edges > 1:
                    sim, t_relly, s_relly = return_closest_relation_pair(t_rels, s_rels)
                    candidate_pair = [[t_subj, t_relly, t_obj], [s_subj, s_relly, s_obj], dist]
                    dud = 0
                    if t_num_edges > 1 and s_num_edges > 1:  # FIXME or or and
                        compatible_edges = add_consistent_multi_edges_to_mapping(t_grf, s_grf, t_subj, t_reln, t_obj,
                                            s_subj, s_reln, s_obj, candidate_pair, globl_mapped_predicates, semantics)
                        if compatible_edges: # FIXME check for duplications first
                            new_list = globl_mapped_predicates + compatible_edges # candidate pair included, as appropriate
                else:
                    new_list = globl_mapped_predicates + [candidate_pair]
                #return explore_mapping_space(t_grf, s_grf, t_preds_list[1:], s_preds_list[1:], new_list)
                return explore_mapping_space(t_grf, s_grf, t_preds_list[1:], s_preds_list[1:], new_list)
        explore_mapping_space(t_grf, s_grf, t_preds_list, s_preds_list[1:], globl_mapped_predicates)  # FIXME 2024 4 05
    #print(" **REACHED HERE**")
    return explore_mapping_space(t_grf, s_grf, t_preds_list[1:], s_preds_list[1:], globl_mapped_predicates)


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
    overall_list = []
    for w1 in lis1:
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
# align_words_by_s2v(['at', 'shoot', 'pledge'], ['give', 'knew', 'saw', 'glided'])


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
                compatible_pairs += [[[t_obj, rel, t_subj], [s_obj, rel, s_subj], 0.0]]  # reverse subj & obj
        dud1 = list_diff(t_rels_1, overlap_1)
        dud2 = list_diff(s_rels_1, overlap_1)
        pairs_1 = align_words_by_s2v(dud1, dud2)
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
# align_words_by_s2v(['give', 'knew', 'saw', 'glided'], ['pledged', 'shoot', 'at'])


def list_diff(li1, li2):
    temp3 = []
    for element in li1:
        if element not in li2:
            temp3.append(element)
    return temp3

def count_occurrences_of_tuple_in_list_UNUSED(tupl_1, tupl_2, list_of_node_pairs):
    count = 0
    for prd in list_of_node_pairs:
        if tupl_1 == prd[0] and tupl_2 == prd[1]:
            count += 1
    return count

def check_if_source_already_mapped(s_pred, globl_mapped_predicates):
    if s_pred == [] or globl_mapped_predicates == []:
        return False
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



def evaluate_mapping(target_graph, source_graph, globl_mapped_predicates, semantics=True):
    """ Generate the mapping dictionary from globl_mapped_predicates.
    [[['hawk_Karla_she', 'saw', 'hunter'], ['hawk_Karla_she', 'know', 'hunter'], 0.715677797794342], [['h"""
    global mapping
    mapping = dict()
    relatio_structural_dist = 0
    paired_relations = []
    rel_s2v, con_s2v, scor = 0, 0, 0
    for t_pred, s_pred, val in globl_mapped_predicates:  # full predicates
        relatio_structural_dist += val
        if t_pred == [] or s_pred == []:
            continue
        elif t_pred[0] in mapping.keys():
            if mapping[t_pred[0]] != s_pred[0]:  # extend the mapping
                print(" Mis-Mapping 1 in DFS ")
                sys.exit(" Mis-Mapping 1 in DFS ")
        else:
            if s_pred[0] not in mapping.values():
                mapping[t_pred[0]] = s_pred[0] # new mapping
            else:
                sys.exit("Mis Mapping 1 b")
        if t_pred[2] in mapping.keys():  # PTNT role
            if mapping[t_pred[2]] != s_pred[2]:  # extend the mapping
                print(t_pred, "    ", s_pred)
                sys.exit(" Mis-Mapping 2 in DFS")
        else:
            if s_pred[2] not in mapping.values():
                mapping[t_pred[2]] = s_pred[2] # new mapping
            else:
                sys.exit(" Mis-Mapping 2 PTNT in DFS ")
        paired_relations.append([t_pred[1], s_pred[1]])
        rel_s2v += relational_distance(t_pred[1], s_pred[1])
    # Extend the mapping
    # print(" Next: mop-up", end=" ")
    count_limit = 0
    while True and count_limit < 3:
        unmapped_target_preds = return_unmapped_target_preds(target_graph, globl_mapped_predicates)
        unmapped_source_preds = return_unmapped_source_preds(source_graph, globl_mapped_predicates)
        extra_mapped_preds, mapping_item_pairs = \
            possible_mop_up_achievable(unmapped_target_preds, unmapped_source_preds, bestEverMapping, mapping)
        if extra_mapped_preds == [] or unmapped_target_preds == [] or unmapped_source_preds == []:
            break
        count_limit += 1
    if semantics:
        for k,v in mapping.items():
            con_s2v += conceptual_distance(k,v)
    return relatio_structural_dist # rel_s2v, con_s2v, len(globl_mapped_predicates)


def return_unmapped_target_preds(target_graph, globl_mapped_predicates):
    unmapped = []
    for s,o,r in target_graph.edges(data=True):
        found = False
        for tgt, src, val in globl_mapped_predicates:
            if not found and s == tgt[0] and o == tgt[2] and (r == tgt[1] or (r == None or tgt[1] == None)):
                found = True
                break
        if found == False:
            if r:
                unmapped.append([s,r['label'],o])
            else:
                unmapped.append([s, None, o])
    return unmapped


def return_unmapped_source_preds(source_graph, globl_mapped_predicates):  # candidate inferences
    unmapped = []
    for s,o,r in source_graph.edges(data=True):
        found = False
        for tgt, src,_ in globl_mapped_predicates:
            if not found and s == src[0] and o == src[2] and (r == src[1] or (r == None or src[1] == None)):
                found = True
                break
        if found == False:
            if r:
                unmapped.append([s,r['label'],o])
            else:
                unmapped.append([s, None, o])
    return unmapped


def possible_mop_up_achievable(t_preds_list, s_preds_list, bestEverMapping, mapped_concepts, try_harder=False):
    """ Opportunistic extension to mapping, caused by narrowness of beam search. """
    mapping_extra, mx2 = [], []
    for t_subj, t_rel, t_obj in t_preds_list:
        for s_subj, s_rel, s_obj in s_preds_list:
            if relational_distance(t_rel, s_rel) > max_relational_distance:
                continue
            elif conceptual_distance(t_subj, s_subj) >= max_conceptual_distance:
                continue
            elif conceptual_distance(t_obj, s_obj) >= max_conceptual_distance:
                continue
            subjects_compatible = (check_if_already_mapped(t_subj, s_subj, bestEverMapping) or
                                   check_if_both_unmapped(t_subj, s_subj, bestEverMapping))
            objects_compatible = (check_if_already_mapped(t_obj, s_obj, bestEverMapping) or
                                  check_if_both_unmapped(t_obj, s_obj))
            if subjects_compatible and objects_compatible:
                mapping_extra.append([[t_subj, t_rel, t_obj], [s_subj, s_rel, s_obj], 1.11])
                bestEverMapping.append([[t_subj, t_rel, t_obj], [s_subj, s_rel, s_obj], 1.11])
                mapped_concepts[t_subj] = s_subj
                mapped_concepts[t_obj] = s_obj
                break
            elif (check_if_already_mapped(t_obj, s_obj, bestEverMapping) or
                  check_if_both_unmapped(t_subj, s_subj, bestEverMapping)):
                mapping_extra += [[[t_subj, t_rel, t_obj], [s_subj, s_rel, s_obj], 1.11]]
                bestEverMapping.append([[t_subj, t_rel, t_obj], [s_subj, s_rel, s_obj], 1.11])
                mapped_concepts[t_subj] = s_subj
            elif try_harder == True and check_if_both_unmapped(t_subj, s_subj, bestEverMapping) \
                    and check_if_both_unmapped(t_obj, s_obj, bestEverMapping) and \
                    relational_distance(t_rel, s_rel) < max_relational_distance:
                mapping_extra += [[[t_subj, t_rel, t_obj], [s_subj, s_rel, s_obj], 1.34]]
                bestEverMapping.append([[t_subj, t_rel, t_obj], [s_subj, s_rel, s_obj], 1.34])  # Sept 23
                mapped_concepts[t_subj] = s_subj
    if mapping_extra != []:
        for tgt, src, val in mapping_extra:
            t_preds_list.remove(tgt)
            s_preds_list.remove(src)
        print(" XXXXXXX Map Extension ", len(mapping_extra), " ++ ", mapping_extra, end="  ")
        # print(" XXXXXXX Map Extension by ", len(mapping_extra), end="  ")
    if try_harder == False:
        mx2, mp2 = possible_mop_up_achievable(t_preds_list, s_preds_list, bestEverMapping, mapped_concepts, try_harder = True)
    if mx2 != []:
        print("EXTRA Extension found ", len(mx2), end = " XX    ")
        mapping_extra = mapping_extra + mx2
    return mapping_extra, mapped_concepts


def euclidean_distance(t_subj_in, t_subj_out, a_to_b, b_to_a, t_obj_in, t_obj_out,  w,x,y,z,
                       s_subj_in, s_subj_out, p_to_q, q_to_p, s_obj_in, s_obj_out, w1,x1,y1,z1):  # new version
    z = math.sqrt((t_subj_in - s_subj_in) ** 2 + (t_subj_out - s_subj_out) ** 2 + (a_to_b - p_to_q) ** 2 +
                  (t_obj_in - s_obj_in) ** 2 + (b_to_a - q_to_p) ** 2 + (t_obj_out - s_obj_out) ** 2 +
                  (w-w1)**2 + (x-x1)**2 + (y-y1)**2 + (z-z1)**2 )
    return z  # + 0.0001


def euclid_dist_2(*args):
    z, rslt = len(args), 0
    if z % 2 != 0:
        sys.exit("error in vector length")
    arg_lis = list(args)
    limit = int(z/2)
    for itm in range(limit):
        rslt += abs(arg_lis[itm] - arg_lis[itm + limit])
    return rslt
# ans = euclid_dist_2(0,0, 1,2,3,4)


def relational_distance(t_in, s_in):  # using s2v sense2vec
    global term_separator
    global s2v_verb_cache
    global s2v
    if t_in == s_in:
        return 0
    elif mode == "Code":  # identical relations only, identicality constraint
        return 1
    # print("REL-Dis:", t_in, s_in, end=" ")
    if isinstance(t_in, dict):
        dud=0
    t_reln = t_in.split(term_separator)[0]
    s_reln = s_in.split(term_separator)[0]
    if t_in == "" or s_in == "" or t_reln == "" or s_reln == "":  # treat this as an error?
        return 0.9876
    if t_reln[0] == s_reln[0]:
        return 0.0001
    elif second_head(t_reln) == second_head(s_reln):
        return 0.001
    elif t_reln + "-" + s_reln in s2v_verb_cache:
        sim_score = s2v_verb_cache[t_reln + "-" + s_reln]
        return sim_score
    elif s_reln + "-" + t_reln in s2v_verb_cache:
        sim_score = s2v_verb_cache[s_reln + "-" + t_reln]
        return sim_score
    else:
        if s2v.get_freq(t_reln + '|VERB') is None or s2v.get_freq(s_reln + '|VERB') is None:
            t_root = wnl.lemmatize(t_reln)
            s_root = wnl.lemmatize(s_reln)
            if s2v.get_freq(t_root + '|VERB') is None or s2v.get_freq(s_root + '|VERB') is None:
                return 1.0
            else:
                sim_score = 1 - s2v.similarity([t_root + '|VERB'], [s_root + '|VERB'])
                s2v_verb_cache[t_reln + "-" + s_reln] = sim_score
                return sim_score
        else:
            sim_score = 1 - s2v.similarity([t_reln + '|VERB'], [s_reln + '|VERB'])
            s2v_verb_cache[t_reln + "-" + s_reln] = sim_score
            return sim_score
# relational_distance("walk", "run") -> 0.5572440028190613


def conceptual_distance(str1, str2):
    """for simple conceptual similarity"""
    global term_separator
    global s2v_noun_cache
    if str1 == str2:
        return 0.0001
    elif isinstance(str1, (int, float)):  # numeric generated graphs
        return abs(str1 - str2)
    #arg1 = str1.split(term_separator)
    #arg2 = str2.split(term_separator)
    arg1 = str1.replace(".", ":").split(term_separator)
    arg2 = str2.replace(".", ":").split(term_separator)
    if mode == "Code":
        if arg1[0] == arg2[0]:
            if len(arg1) >1 and len(arg2)>2 and arg1[1] == arg2[1]:
                inter_sectn = list(set(arg1) & set(arg2))
                if len(inter_sectn) > 0:
                    return min(0.1, 0.5 / len(inter_sectn))
                else:
                    return 0.1
            else:
                return 0.05
        else:
            return 1
    elif mode == "English":
        if arg1[0] + "-" + arg2[0] in s2v_verb_cache:
            sim_score = s2v_verb_cache[arg1[0] + "-" + arg2[0]]
            return sim_score
        elif arg2[0] + "-" + arg1[0] in s2v_verb_cache:
            sim_score = s2v_verb_cache[arg2[0] + "-" + arg1[0]]
            return sim_score
        elif s2v.get_freq(arg1[0] + '|NOUN') is None:
            if s2v.get_freq(arg2[0] + '|NOUN') is None:
                return 0.2
            else:
                return 1.115
        elif s2v.get_freq(arg2[0] + '|NOUN') is not None:
            sim_score = s2v.similarity([arg1[0] + '|NOUN'], [arg2[0] + '|NOUN'])
            s2v_noun_cache[arg1[0] + "-" + arg2[0]] = sim_score
            return sim_score
        else:
            return 1.11
    else:
        return 1.116
# conceptual_Distance("car", "bus") -> 0.51187253

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def return_sorted_predicates(grf):  # 6 terms plus PageRank
    """ subj_in, subj_out, s_o, o_s, obj_in, obj_out, sub_pr, obj_pr, sub_sub, obj_obj, S,V,O  """
    edge_list = grf.edges.data("label")
    pred_list = []
    pr = nx.pagerank(grf, alpha=0.85)
    # centrality = nx.degree_centrality(grf)
    #pr_rev = nx.pagerank(grf_rev, alpha=0.85)
    #pr_lis = [round(x,4) for x in list(pr.values())]
    #pr_rev_lis = [round(x, 4) for x in list(pr_rev.values())]
    #print("\nPR  ", pr_lis[0:15], "\nPR2 ", pr_rev_lis[0:15])
    for (s, o, v) in edge_list:
        a = grf.number_of_edges(s, o)
        b = grf.number_of_edges(o, s)
        y = grf.number_of_edges(s, s)
        z = grf.number_of_edges(o, o)
        w = math.sqrt(pr[s])
        x = math.sqrt(pr[o])
        pred_list.append([grf.in_degree(s), grf.out_degree(s), a, b, grf.in_degree(o), grf.out_degree(o),
                          w, x, y, z, s, v, o])
    z = sorted(pred_list, key=my_sub_sort, reverse=True)  # total sum(lst[0:10])
    return z


def my_sub_sort(lst):
    digits = sum(lst[0:10])
    return digits


def return_edges_relations_UNUSED(G):  # returnEdges(sourceGraph)
    """returns a list of edge names, followed by a printable string """
    res = ""
    for (u, v, reln) in G.edges.data('label'):
        res = res + u + " " + reln + " " + v + '.' + "\n"
        #print(reln, end=" ")
    return res


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


def addRelationsToMapping_DEPRECATED(target_graph, source_graph, mapping_dict, s_decoding, t_decoding):  # LCS_Number
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


def return_list_of_mapped_concepts_DEPRECATED(list_of_mapped_preds):
    """ Return a dictionary of paired concepts/nouns. No tests conducted. """
    mapping_dict = {}
    for t_pred, s_pred, val in list_of_mapped_preds:
        mapping_dict[t_pred[0]] = s_pred[0]
        mapping_dict[t_pred[2]] = s_pred[2]
    res = list(mapping_dict.items())
    return res



