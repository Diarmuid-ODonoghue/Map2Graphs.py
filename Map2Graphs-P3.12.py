class MappingObject:
    def __init__(self):
        self.mapping = {}
        self.mapping['Total_Score'] = 0
        self.mapping['Number_Mapped_Predicates'] = 0

import networkx as nx
# import isomorphvf2CB  # VF2 My personal Modified VF2 variant
import csv
# import pprint
import numpy  # as np
import os
import errno
import sys
import time
# import HOGS
import HOGS2
import HOGS
import ShowGraphs
# from nltk.corpus import wordnet
# from loguru import logger, snoop, heartrate
# from itertools import count, product
# import multiprocessing   # from multiprocessing import Process, Queue, Manager
# import subprocess
# import ConceptNetElaboration as CN
# import pylab, operator
# from cacheout import Cache
# import requests, pyvis

import nltk # Python_3.12 edits
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag  # Python_3.12 edits
#nltk.download('punkt_tab')
#nltk.download('averaged_perceptron_tagger_eng')


if True: # Flase for source code operation
    from sense2vec import Sense2Vec
    # s2v = Sense2Vec().from_disk("C:/Users/user/Documents/Python-Me/Sense2Vec/s2v_reddit_2019_lg/")
    s2v = Sense2Vec().from_disk("C:/Users/dodonoghue/Documents/Python-Me/Sense2Vec/s2v_reddit_2019_lg/")
    query = "drive|VERB"
    assert query in s2v
    print("S2v similarity value", s2v.get_freq(query))

# nx.MultiDiGraph()  # loosely-ordered, mostly-directed, somewhat-multiGraph, self-loops, parallel edges

global localBranch
global max_graph_size
global algorithm
global perform_eager_concept_fusion
global mapping_graph
global unmapped_target_edges
global list_of_target_preds  # used to communicate unmapped target predicates
global generate_inferences
global coalescing_completed
global list_of_mapped_preds
# global mapping_run_time
global analogyFilewriter
global semantics
global rel_s2v
global rel_count
global con_s2v
global con_count

algorithm = "HOGS2"
# algorithm = "VF2++"
# algorithm = "ismags"
# algorithm = "VF2"

base_path = "C:/Users/user/Documents/Python-Me/data"
base_path = "C:/Users/dodonoghue/Documents/Python-Me/data"
# basePath = "C:/Users/dodonoghue/Documents/Python-Me/Python2Graph"
# basePath = dir_path = os.path.dirname(os.path.realpath(__file__)).replace('\\','/')

if True:
    file_type_filter = ".csv"
    identical_edges_only = True
    mode = 'Code'
    generate_inferences = False
    max_graph_size = 200  # see prune_peripheral_nodes(graph) pruning
    skip_over_previous_results = False  # redo, repeat,
    term_separator = ":"  # Block:Else: If
    semantics = False
    show_blended_graph = False
    show_input_graph = False
    base_path = "C:/Users/dodonoghue/Documents/Python-Me/"
    # localBranch = "/c-sharp-id-num_selection/"
    # localBranch = "/py-Csvs/"
    # localBranch = "/AllCSVs/"
    # localBranch = "/C-Sharp Data/"
    # localBranch = "/Java Data/"
    # localBranch = "/test/"
    # localBranch = "/Java Data/"
    # localBranch = "Python2Graph/output-graphs/"
    localBranch = "data/Python data/" # localBranch = "/Python data/"
else:
    file_type_filter = ".RVB" # Aln .csv ".OpIE" # "RVB"  # ".dcorf.csv" # RVB"  # filename ends with
    from nltk.corpus import wordnet_ic
    from nltk.corpus import wordnet as wn
    from nltk.corpus import words as nltkwords
    from nltk.stem import WordNetLemmatizer
    import nltk
    identical_edges_only = False
    #if True:  # False enables quick testing
    # semcor_ic =
    brown_ic = wordnet_ic.ic('ic-brown.dat') # wordnet_ic.ic('ic-semcor.dat')
    wnl = WordNetLemmatizer()
    mode = 'English'
    skip_over_previous_results = False  # redo, repeat,
    perform_eager_concept_fusion = False  # fuse concepts
    show_blended_graph = False
    show_input_graph = False
    generate_inferences = True
    max_graph_size = 250    # see prune_peripheral_nodes(graph) pruning
    term_separator = "_"   # "_"  hawk_he
    skip_prepositions = True
    semantics = True
    # localBranch = "/test/"  # "test/"
    # localBranch = "/iProva/"
    # localBranch = "/2020 Covid-19/"
    # localBranch = "/Psychology data/" #GPT3-5 Generated Sources/"
    # localBranch = "/ICCC Corpus/" #GPT3-5 Generated Sources/"
    localBranch = "/Aesops Fables/"
    # localBranch = "/MisTranslation Data/"
    # localBranch = "/Killians Summaries/"
    # localBranch = "/SIGGRAPH ROS - Dr Inventor/"
    # localBranch = "/Sheffield-Plagiarism-Corpus/"
    # localBranch = "/20 SIGGRAPH Abstracts - Stanford/"
    # localBranch = "/SIGGRAPH csv - Dr Inventor/"
    # localBranch = "/Microsoft Paraphrase Corpus/"
    # localBranch = "/Requirements - VBZ/"
    # localBranch = "/21 Karp NP/"

# ######################################################
# ############## Global Variables ######################
# ######################################################

target_graph = nx.MultiDiGraph()  # <- targetFile
source_graph = nx.MultiDiGraph()  # <- sourceFile
temp_graph2 = nx.MultiDiGraph()  # create ordered graphs. Subsequently treat as unordered.
GM = dict()

list_of_inferences = []
LCSLlist = []
list_of_mapped_preds = []
relationMapping = []
WN_cache = {}
CN_dict = {}


localPath = base_path + localBranch
htmlBranch = base_path + localBranch + "FDG/"

#pp = pprint.PrettyPrinter(indent=4)

# #######################################################
# ############# File Infrastructure #####################
# #######################################################

CN_file_name = "C:/Users/dodonoghue/Documents/Python-Me/data" + "/ConceptNetdata.csv"
CSVPath = localPath + "/" + "ResultsOutput/"  # Where you want the CSV file to be produced
CachePath = base_path + localBranch + "/Cache.txt"  # Where you saved the Cache txt file
analogyFileName = "temporary.csv"
list_of_mapped_preds = []

print("\nINPUT:", localPath, end=" ")
print("\nOUTPUT:", CSVPath)


#localPath = 'C:/Users/dodonoghue/Documents/Python-Me/data/c-sharp-id-num_selection/'
source_files = os.listdir(localPath)
all_csv_files = [i for i in source_files if i.endswith(file_type_filter)]  # if ("code" in i) and (i.endswith('.csv'))]
all_csv_files.sort(reverse=True)
print(len(all_csv_files), " files including; ", all_csv_files[0:4], end="... etc")
print("Mode=", mode, "  Term Separator=", term_separator)
print("# CSV input files in: ", CSVPath, "=", len(all_csv_files))

commutative_verb_list = ['and', 'or', 'beside', 'near', 'next_to']  # x and y  ==>  y and x

# pronouns and pronomial adjectives
pronoun_list = ["all", "another", "any", "anybody", "anyone", "anything", "as", "aught", "both", "each",
                "each other", "either", "enough", "everybody", "everyone", "everything", "few", "he", "her",
                "hers", "herself", "him", "himself", "his", "I", "idem", "it", "its", "itself", "many", "me ",
                " mine", "most", "my", "myself", "naught", "neither", "no one", "nobody", "none", "nothing",
                "nought", "one", "one another", "other", "others", "ought", "our", "ours", "ourself",
                "ourselves", "several", "she", "some", "somebody", "someone", "something", "somewhat",
                "such", "suchlike", "that", "thee", "their", "theirs", "theirself", "theirselves", "them",
                "themself", "themselves", "there", "these", "they", "thine", "this", "those", "thou", "thy",
                "thyself", "us", "we ", " what", "whatever", "whatnot", "whatsoever", "whence", "where",
                "whereby", "wherefrom", "wherein", "whereinto", "whereof", "whereon", "wherever", "wheresoever",
                "whereto", "whereunto", "wherewith", "wherewithal", "whether", "which", "whichever",
                "whichsoever", "who", "whoever", "whom", "whomever", "whomso", "whomsoever", "whose",
                "whosever", "whosesoever", "whoso", "whosoever", "ye", "yon", "yonder", "you", "your", "yours",
                "yourself", "yourselves"]


def preposition_test(word):
    prep_list = ['above', 'across', 'against', 'along', 'among', 'around', 'as', 'at', 'before', 'behind', 'below',
                 'beneath', 'beside', 'between', 'by', 'down', 'from', 'for', 'in', 'into', 'of', 'off',
                 'on', 'than', 'through', 'to', 'toward', 'under', 'upon', 'with', 'within']  # 'near'
    return word in prep_list


# #################################################################################################################
# ##################################### Process Input #############################################################
# #################################################################################################################

def extend_as_set(l1, l2):
    result = []
    if len(l1) >= len(l2):
        result.extend(x for x in l1 if x not in result)
        donor = l2
    else:
        result.extend(x for x in l2 if x not in result)
        donor = l1
    result.extend(x for x in donor if x not in result)
    coref_terms = '_'.join(word for word in result)
    return reorganise_coref_chain(coref_terms)
# extend_as_set(['hunter', 'He'], ['hunter', 'he', 'him'])


concept_tags = {'NN', 'NNS', 'PRP', 'PRP$', 'NNP', 'NNPS'}

def trim_concept_chain(text):  # for long chains only, phrases
    """Extract nouns and Preps from coreferents that are entire phrases."""
    str = nltk.word_tokenize(text.replace("_", " "))
    tagged = nltk.pos_tag(str)
    ret = '_'.join([word for word, tag in tagged[:-1] if tag in concept_tags] + [tagged[-1][0]])
    return ret
# trim_concept_chain('cloth_captured_John_Americans_from_a_flapping_flag_it')


def my_flatten_list(nested_list):
    flat_list = []
    if len(nested_list) == 0:
        return nested_list
    for sublist in nested_list:
        #for item in sublist:
        flat_list.append(sublist)
    return flat_list


def string_from_sub_lists(lis):
    ret = ""
    for sub in lis:
        ret += sub
    return ret
# z = ['C:\\Users\\user\\Documents\\Python-Me\\data\\Psychology data\\01 Antonietti - Four-Canals-CarocciB1.S.txt\t3\tthe canal\tmight damage\tthe surrounding areas\t17\t19\t24\t26\t26\t29\t0.6714262017029996\tHowever ', ' a mason pointed out that during the flood periods the stream of water flowing along the canal might be too strong and might damage the surrounding areas ; by contrast ', ' during the drought periods a unique stream of water might be insufficient to feed the lake .\tRB ', ' DT NN VBD RP IN IN DT NN NNS DT NN IN NN VBG IN DT NN MD VB RB JJ CC MD VB DT VBG NNS : IN NN ', ' IN DT NN NNS DT JJ NN IN NN MD VB JJ TO VB DT NN .\tB-ADVP O B-NP I-NP B-VP B-PRT B-SBAR B-PP B-NP I-NP I-NP B-NP I-NP I-NP I-NP I-NP B-PP B-NP I-NP B-VP I-VP B-ADJP I-ADJP O B-VP I-VP B-NP I-NP I-NP O B-PP B-NP O B-PP B-NP I-NP I-NP B-NP I-NP I-NP I-NP I-NP B-VP I-VP B-ADJP B-VP I-VP B-NP I-NP O\tthe canal\tdamage\tthe surrounding areas']


def build_graph_from_csv(file_name):
    """ Includes eager concept fusion rules. Enforces  noun_properNoun_pronoun"""
    global temp_graph2  # an ORDERED multi Di Graph, converted at the end
    global term_separator
    global perform_eager_concept_fusion
    fullPath = localPath + "/" + file_name
    file_type = file_name.split(".")[-1]
    # set_term_separator(file_type)
    with (open(fullPath, 'r') as csvfile):
        if file_type == "Aln":
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        else:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        temp_graph2.clear()
        temp_graph2.graph['Graphid'] = file_name
        try:
            previous_subj, last_v, previous_obj = "", "", ""
            for row in csvreader:
                row_len = len(row)
                if file_name.endswith(".RVB"):  # ReVerb
                    row_len = 18
                    row = string_from_sub_lists(row)
                    row = row.split('\t')
                match row_len:
                    case 3:
                        noun1 = row[0].replace(" ","_")
                        verb = row[1].replace(" ","_")
                        noun2  = row[2].replace(" ","_")
                    case 4:
                        methodName, noun1, verb, noun2 = row
                    case 0:
                        continue
                    case 1:  # just a header
                        continue
                    case 18:  # ReVerb .RVB
                        _,_,noun1, verb, noun2, _,_,_,_,_,_,_,_,_,_,_,_,_ = row
                        noun1 = noun1.replace(" ", term_separator)
                        verb = verb.replace(" ", term_separator)
                        noun2 = noun2.replace(" ", term_separator)
                    case 6:
                        if row[0] == "CodeContracts":  # skip the contracts
                            pass  # break
                        a, noun1, rel, noun2, c, d = row
                        noun1 = noun1.strip() #+ term_separator + a
                        noun2 = noun2.strip() #+ term_separator + b
                        verb = rel
                    case 6:
                        if row[0] == "CodeContracts":  # skip the contracts
                            pass  # break
                        methodName, noun1, verb, noun2, num1, num2 = row
                        noun1 = noun1.strip() + term_separator + num1
                        noun2 = noun2.strip() + term_separator + num2
                    case 5:
                        useful, sentencea, sentenceb, pos1, pos2 = row
                        noun1 = useful.split('\t')[2]
                        verb = useful.split('\t')[3]
                        noun2 = useful.split('\t')[4]
                    case _:
                        print("\nUnexpected row length: ", row_len, ">>", row, "<<")
                        continue
                noun1 = noun1.strip().lower()   # remove spaces
                verb = verb.strip().lower()
                noun2 = noun2.strip().lower()
                if mode == 'English':
                    if skip_prepositions and preposition_test(verb):
                        continue
                    if (noun1 == previous_subj) and (noun2 == previous_obj) and \
                            not (is_verb(verb)) and preposition_test(verb):
                        verb = last_v + "_" + verb
                        print(verb, end=" ")
                        temp_graph2.remove_edge(noun1, noun2)
                    elif noun1 == "NOUN":  # skip header information
                        continue

                    if len(noun1.split(term_separator)) > 1:
                        noun1 = parse_new_coref_chain(noun1)
                    if len(noun2.split(term_separator)) > 1:
                        noun2 = parse_new_coref_chain(noun2)

                temp_graph2.add_node(noun1, label=noun1)
                temp_graph2.add_node(noun2, label=noun2)
                temp_graph2.add_edge(noun1, noun2, label=verb, key=verb)  # added key=verb

                if mode == 'English':       # phrasal verbs; read_over, read_up, read_out
                    previous_subj = noun1
                    last_v = verb
                    previous_obj = noun2
                # print(temp_graph2.number_of_edges())
        except csv.Error as e:
            sys.exit('file %s, line %d: %s' % (CSVPath, csvreader.line_num, e))
    if mode == 'English' and perform_eager_concept_fusion:
        tmp1 = list(temp_graph2.edges(data=True))
        eager_concept_fusion_2(temp_graph2)
        tmp2 = list(temp_graph2.edges(data=True))
        if not len(tmp1) == len(tmp2):
            exit("Flip dud - Merge Error")  # Feck, a merge error
        # print("ECF2 ", tmp, " -> ", temp_graph2.number_of_nodes(), end="")
    # findMissingConnections(temp_graph2)
    returnGraph = nx.MultiDiGraph(temp_graph2)   # no need for a .copy()
    return returnGraph  # return canonical  version of the graph :-)


def set_term_separator(file_type):
    global term_separator
    match file_type:
        case ".csv":
            term_separator = "_"
        case _: # Aln RVB
            term_separator = " "

###################################################



def parse_new_coref_chain(in_chain):
    """ For silly long coref chains. """
    if mode == 'Code':
        return in_chain
    cnt = in_chain.find(term_separator)
    if cnt < 0:
        return in_chain
    else:   # cnt <= 100:  # parser works poorly on short noun sequences
        return reorganise_coref_chain(in_chain)
    return "ERROR - parse_new_coref_chain() "


def reorganise_coref_chain(strg):  # noun-propernoun-pronoun   #w, tag = nltk.pos_tag([wrd])[0]
    global term_separator
    noun_lis, propN_lis, pron_lis, possible_propN_lis = [], [], [], []
    noun_lis2, propN_lis2, pron_lis2, possible_propN_lis2 = [], [], [], []
    if strg.find(term_separator) < 0:
        slt = strg
    else:
        chan = strg.replace(term_separator, " ")
        text = nltk.word_tokenize(chan)
        pos_tag_list = nltk.pos_tag(text)
        for w, tokn in pos_tag_list:
            if w in ['a', 'the', 'its', 'A', 'The', 'Its']:  # remove problematic words
                continue
            elif tokn in ["PRON", "PRP", "PRP$"]:
                pron_lis.append(w)
                pron_lis2.append([w, tokn])
            elif tokn in ["N", "NN", "NNS"]:
                noun_lis.append(w)
                noun_lis2.append([w, tokn])
            elif tokn in ["NNP", "NNPS"]:
                propN_lis.append(w)
                propN_lis2.append([w, tokn])
            else:
                possible_propN_lis.append(w)
                possible_propN_lis2.append([w, tokn])
        noun_lis.reverse()
        alt = list(dict.fromkeys(noun_lis + propN_lis + possible_propN_lis + pron_lis)) # & remove duplicates
        slt = "_".join(noun_lis + propN_lis + possible_propN_lis + pron_lis)
    return slt
# reorganise_coref_chain("its_warlike_neighbor_Gagrach")
# reorganise_coref_chain("He_hunter_he_him")
# reorganise_coref_chain('hawk_she_Karla')
# reorganise_coref_chain('hawk_Karla_she')
# reorganise_coref_chain('parent\'s_house')  # posessives


def flatten_list(t):   # single level of nesting removed
    return [item for sublist in t for item in sublist]


def slightly_flatten(preds_list):
    if preds_list == []:
        return []
    else:
        rslt = []
        for entry in preds_list:
            rslt.append(entry[0] + entry[1])
    return rslt


def head(node):
    global term_separator
    if not (isinstance(node, str)):
        return ""
    else:
        lis = node.split(term_separator)
        if len(lis) >= 1:
            wrd = lis[0].strip()
        #else:
        #    wrd = lis[0].strip()
        return wrd


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


def eager_concept_merge_rules(node_list):
    coalescing_completed = True
    merge_condition_detected = False
    list_of_removed_nodes = []
    nodes_to_be_merged = []
    for graph_node in node_list:  # node_list:  # Final-pass Coalescing
        # gn_n = contains_noun(graph_node)
        gn_pn = contains_proper_noun2(graph_node)
        if graph_node in list_of_removed_nodes:
            continue
        if "_" in graph_node:
            new_node_name = reorganise_coref_chain(graph_node)
            if new_node_name != graph_node:
                graph_node = new_node_name
                # graph_node_reogrganised = True
        gn_word_list = graph_node.split(term_separator)
        extendedNoun = ""
        for graph_node2 in node_list[1 + node_list.index(graph_node):]:  # subsequent nodes only
            if graph_node is graph_node2 or graph_node2 in list_of_removed_nodes:
                continue
            gn2_pn = contains_proper_noun2(graph_node2)
            gn2_word_list = graph_node2.split(term_separator)
            intersect = intersection(gn_word_list, gn2_word_list)
            if len(intersect) > 0 and contains_noun("_".join(intersect)):
                merge_condition_detected = True
            if not merge_condition_detected and gn_pn and gn2_pn and intersection(gn_pn, gn2_pn):
                merge_condition_detected = True
            elif not merge_condition_detected and head_word(graph_node) == head_word(graph_node2):
                r = intersecting_proper_nouns(gn_pn, gn2_pn)
                if r:  # and not different proper nouns, or incompatible pronouns
                    merge_condition_detected = True
            if merge_condition_detected and not (graph_node == graph_node2) and not graph_node in list_of_removed_nodes \
                    and not graph_node2 in list_of_removed_nodes:
                if extendedNoun == "":
                    extendedNoun = extend_as_set(graph_node.split(term_separator), graph_node2.split(term_separator))
                else:
                    extendedNoun = extend_as_set(graph_node.split(term_separator), graph_node2.split(term_separator))
                extendedNoun = reorganise_coref_chain(extendedNoun)
                if not graph_node2 == extendedNoun:
                    nodes_to_be_merged.append([graph_node2, extendedNoun])
                    list_of_removed_nodes.append(graph_node2)
                if not (graph_node == extendedNoun):
                    nodes_to_be_merged.append([graph_node, extendedNoun])
                    list_of_removed_nodes.append(graph_node)
                merge_condition_detected = False
                # coalescing_completed = False
    return nodes_to_be_merged   # ordered


def return_coref_nodes_only(nod_list):
    res = []
    for n in nod_list:
        if isinstance(n, str) and term_separator in n:
            res += [n]
    return res


def return_eager_merge_candidates(inGraph):
    global term_separator
    global coalescing_completed
    global temp_graph2
    node_list = list(inGraph.nodes())
    coref_nodes = return_coref_nodes_only(node_list)
    coalescing_completed = True       # ##########################################################################
    merge_condition_detected = False
    list_of_removed_nodes = []
    nodes_to_be_merged = []
    reorganise_condition_detected = False
    for graph_node in node_list:  # eager_concept_merge_rules(node_list)
        if not isinstance(graph_node, str):
            continue
        gn_pn = contains_proper_noun2(graph_node)
        if graph_node in list_of_removed_nodes:
            continue
        if isinstance(graph_node, str) and "_" in graph_node:
            new_node_name = reorganise_coref_chain(graph_node)
            if new_node_name != graph_node:
                reorganise_condition_detected = True
        gn_word_list = graph_node.split(term_separator)
        extendedNoun = ""
        graph_node_low = graph_node.lower().split(term_separator)
        for graph_node2 in coref_nodes:  # node_list[1 + node_list.index(graph_node):]:  # coref_nodes
            if graph_node is graph_node2: # or graph_node2 in list_of_removed_nodes:
                continue
            elif graph_node2 in list_of_removed_nodes:
                continue
            gn2_pn = contains_proper_noun2(graph_node2)
            gn2_word_list = graph_node2.split(term_separator)
            candidate_intersect = intersection(graph_node_low, graph_node2.lower().split(term_separator))
            # if candidate_intersect != []:
            intersect = []
            for w in candidate_intersect:  # remove pronouns from intersection
                if not is_pronoun(w): # and regular nouns?
                    for z in graph_node2:
                        if z.lower() in candidate_intersect:
                            intersect += [z]
            residue1 = set(gn_pn) - set(intersect)
            residue2 = set(gn2_pn) - set(intersect)
            if len(intersect) > 0 and residue1 =={} and residue2=={}: # contains_noun("_".join(intersect)):
                merge_condition_detected = True
            if not merge_condition_detected and gn_pn and gn2_pn and len(intersection(gn_pn, gn2_pn))>0 and\
                    not (len(residue1)>0 and len(residue2)>0):
                merge_condition_detected = True
            elif not merge_condition_detected and head_word(graph_node) == head_word(graph_node2):
                r = intersecting_proper_nouns(gn_pn, gn2_pn)
                if r:  # and not different proper nouns, or incompatible pronouns
                    merge_condition_detected = True
            elif not merge_condition_detected and head_word(graph_node_low) == head_word(graph_node2.lower()):
                merge_condition_detected = True
            # ### Perform merge
            if merge_condition_detected and not (graph_node == graph_node2) and not graph_node in list_of_removed_nodes \
                    and not graph_node2 in list_of_removed_nodes:
                if extendedNoun == "":
                    extendedNoun = extend_as_set(graph_node.split(term_separator), graph_node2.split(term_separator))
                else:
                    extendedNoun = extend_as_set(graph_node.split(term_separator), graph_node2.split(term_separator))
                extendedNoun = reorganise_coref_chain(extendedNoun)
                if not graph_node2 == extendedNoun:
                    nodes_to_be_merged.append([graph_node2, extendedNoun])
                    list_of_removed_nodes.append(graph_node2)
                if not (graph_node == extendedNoun):
                    nodes_to_be_merged.append([graph_node, extendedNoun])
                    list_of_removed_nodes.append(graph_node)
            if reorganise_condition_detected and not merge_condition_detected:
                nodes_to_be_merged.append([graph_node, new_node_name])
                list_of_removed_nodes.append(graph_node)
                reorganise_condition_detected = False
            merge_condition_detected = False
    return nodes_to_be_merged


def eager_concept_fusion_2(in_graph):  # EAGER CONCEPT FUSION of parsed nodes
    global temp_graph2
    nodes_to_be_merged = "dummy"
    iter_count = 0
    original_number_of_nodes = in_graph.number_of_nodes()
    original_number_of_edges = in_graph.number_of_edges()
    while nodes_to_be_merged != [] and iter_count <= 1:  # double repetition of cascading
        nodes_to_be_merged = return_eager_merge_candidates(in_graph)
        for graph_node, extendedNoun in nodes_to_be_merged:     # execute concept merge
            if not in_graph.has_node(graph_node):
                continue
            print(" MERGE ", graph_node, "->", extendedNoun, end="--    ")
            # FIXME: contracted_nodes() seems to duplicate self-edges
            num1 = in_graph.number_of_edges()  # list(in_graph.edges.data())
            before_edge_list = sorted(list(in_graph.edges.data()), key=lambda tripl: tripl[2]['label'])
            if graph_node in in_graph.nodes(): #  and extendedNoun in in_graph.nodes():
                if in_graph.has_edge(graph_node, graph_node): # possibly worrying because of NetworkX error
                    dup_edge_list = []
                    for (u, v, c) in in_graph.edges.data('label'):
                        if u == graph_node and v == graph_node:
                            dup_edge_list += [c]
                nx.contracted_nodes(in_graph, extendedNoun, graph_node, self_loops=True, copy=False)
                after_edge_list = sorted(list(in_graph.edges.data()), key=lambda tripl: tripl[2]['label'])
            num2 = in_graph.number_of_edges()
            if num1 != num2:
                for n in dup_edge_list:
                    if in_graph.has_edge(extendedNoun, extendedNoun):
                        in_graph.remove_edge(extendedNoun, extendedNoun)
                print("ECF2 Merge ERROR! - repair attempted.", end=" ")
                if in_graph.number_of_edges() == len(before_edge_list):
                    print(" Seemingly successful Repair")
                else:
                    print(" Repair unsuccessful.")  # exit("highly problematic node merging")
            #     remapping = {graph_node: extendedNoun}
            #     in_graph = nx.relabel_nodes(in_graph, remapping, copy=False)
            #     in_graph.nodes[extendedNoun]['label'] = extendedNoun
        iter_count += 1
    num_removed_nodes = original_number_of_nodes - in_graph.number_of_nodes()
    if original_number_of_nodes - in_graph.number_of_nodes() > 0:
        print(" ECF2 reduction by ", num_removed_nodes, end=" nodes.  ")
    final_number_of_edges = in_graph.number_of_edges()
    if original_number_of_edges != final_number_of_edges:
        exit("ECF2 Error - edge deletion during merge.")
    temp_graph2 = in_graph.copy()
    return


def intersecting_proper_nouns(list1, list2):
    if list1 and list2:
        return intersection(list1, list2)
    elif not list1 == False and not list2 == False:
        return True
    else:
        return False


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def list_diff(li1, li2):
    return (list(set(li1) - set(li2)))


# ####################################################################################
# ####################################################################################
# ####################################################################################


def head_word(term):  # first word before term separator
    global term_separator
    if type(term) is list:
        z = term[0].split(term_separator)[0]
    else:
        z = term.split(term_separator)[0]
    return z


def contains_noun(chain):  # noun... PropN ... Pron
    if not(chain):
        return False
    global term_separator
    propN_lis = []
    wrd_lis = chain.split(term_separator)
    pos_tag_list = nltk.pos_tag(wrd_lis)
    if chain == 'bulk':
        print(chain, end="--- ")
    indx = 0
    while indx < len(wrd_lis) and (wn.synsets(wrd_lis[indx], pos=wn.NOUN)
                                   or pos_tag_list[indx][1] in ['N', 'NN']):   # common nouns
        propN_lis.append(wrd_lis[indx])
        indx += 1
    if propN_lis == []:
        return []
    else:
        return propN_lis


def is_noun(wrd):
    zz = wn.synsets(wrd, pos=wn.NOUN)
    return zz != []


def is_proper_noun(wrd):
    w, tag = nltk.pos_tag([wrd])[0]
    return tag == "NNP" or tag == "NP" # NN, NNS, NNP, NNPS
# is_proper_noun("Karla")   is_proper_noun("karla")
# print(is_proper_noun("Timmy Murphy"))
# print(is_proper_noun("aardvark"))
# print(is_proper_noun("Tom O'Sullivan"))
# print(is_proper_noun("karla"))
# print(is_proper_noun("Bezos"))
# print(is_proper_noun("Timmy Murphy"))


def is_verb(wrd):
    #from nltk.tag import pos_tag  # Python_3.12 edits
    #from nltk.tokenize import word_tokenize
    #nltk.download('averaged_perceptron_tagger_eng')
    tokn = nltk.word_tokenize(wrd)
    w, tag = nltk.pos_tag(tokn)[0]
    return tag in ["VB", "VBZ", "VBN", "VBG", "VBD"]
# is_verb("run")
# stop()

def contains_proper_noun(chain):  # noun... PropN ... Pron
    """Returns 1st ProperNoun from the input chain, using NLTK"""
    global term_separator
    propN_lis = []
    wrd_lis = chain.split(term_separator)
    pos_tag_list = nltk.pos_tag(wrd_lis)
    indx = 0
    while indx < len(wrd_lis) and (wn.synsets(wrd_lis[indx], pos=wn.NOUN)
                                   or pos_tag_list[indx][1] in ['N', 'NN']):   # common nouns
        indx += 1
    while indx < len(wrd_lis) and pos_tag_list[indx][1] in ['NNP', 'NP']:    # Proper nouns in the middle
        propN_lis.append(pos_tag_list[indx])
        print("PropN:", pos_tag_list[indx][0], end=" ")
        indx += 1
    if propN_lis == []:
        return []
    else:
        return propN_lis[0][0]
# contains_proper_noun("Karla")   contains_proper_noun("she_karla")
# contains_proper_noun("We_John") contains_proper_noun("It_Boeing")
# contains_proper_noun("We_John_Johns")


def contains_proper_noun2(strg):  # parsing reorganzied chains Sucks!
    if not isinstance(strg, str):
        return []
    chan = strg.replace(term_separator, " ")
    text = nltk.word_tokenize(chan)
    pos_tag_list = nltk.pos_tag(text)
    propN_list = []
    for w, tag in pos_tag_list:
        if tag in ['NP', 'NNP']:
            if w not in propN_list:
                propN_list.append(w)
    if propN_list == []:
        return []
    else:
        return propN_list
# contains_proper_noun("Karla")   contains_proper_noun("she_karla")


def contains_proper_noun_from_lis(lis):  # wrd may be a coreference chain
    global term_separator
    for wo in lis:
        if (not nltkwords.words().__contains__(wo)) and (wn.synsets(wo) == []):
            return wo
    return False


def is_pronoun(wrd):
    return wrd in pronoun_list
# is_pronoun("he")


# ###############################
# ####### Process Graphs ########
# ###############################


def returnEdges(G):  # returnEdges(sourceGraph)
    """returns a list of edge names, followed by a printable string """
    res = ""
    for (u, v, reln) in G.edges.data('label'):
        res = res + u + " " + reln + " " + v + '.' + "\n"
        print(reln, end=" ")
    return res
# returnEdges(targetGraph)


def returnEdgesAsList(G):  # returnEdgesAsList(sourceGraph)
    """ returns a list of lists, each composed of triples"""
    res = []
    for (u, v, reln) in G.edges.data('label'):
        res.append([u, reln, v])
    return res
# returnEdgesAsList(targetGraph)


def returnEdgesBetweenTheseObjects(subj, obj, thisGraph):
    """ returns a list of verbs (directed link labels) between objects - or else [] """
    res = []
    for (s, o, relation) in thisGraph.edges.data('label'):
        if (s == subj) and (o == obj):
            res.append(relation)
    return res


def returnEdgesBetweenTheseObjects_predList(subj, obj, pred_list):
    """ returns a list of verbs (directed link labels) between objects - or else [] """
    res = []
    for (s, v, o) in pred_list:
        if (s == subj) and (o == obj):
            res.append(v)
    return res


def predExists(subj, rel, obj, thisGraph):  # predExists('man','drive','car', sourceGraph)
    for (s, o, r) in thisGraph.edges.data('label'):
        if (s == subj) and (o == obj) and (r == rel):
            return True
    return False


def return_ratio_of_mapped_target_predicates(tgt):  # returnMappingRatio(sourceGraph)
    number_mapped_preds = number_unmapped_preds = 0
    lis = returnEdgesAsList(tgt)
    for (s, v, o) in lis:
        if (s in GM.mapping.keys()) and (v in GM.mapping.keys()) and (o in GM.mapping.keys()):
            number_mapped_preds += 1
        else:
            number_unmapped_preds += 1
    tmp = number_mapped_preds + number_unmapped_preds
    if tmp == 0:
        rslt = 0
    else:
        rslt = number_mapped_preds / tmp
    return number_mapped_preds, rslt


def return_ratio_of_mapped_source_predicates(tgt):  # returnMappingRatio(sourceGraph)
    global GM
    number_mapped_preds, number_unmapped_preds = 0, 0
    lis = returnEdgesAsList(tgt)
    for (s, v, o) in lis:
        if (s in GM.mapping.values()) and (v in GM.mapping.values()) and (o in GM.mapping.values()):
            number_mapped_preds += 1
        else:
            number_unmapped_preds += 1
    tmp = number_mapped_preds + number_unmapped_preds
    if tmp == 0:
        rslt = 0
    else:
        rslt = number_mapped_preds / tmp
    return number_mapped_preds, rslt


def printMappedPredicates(t_graf):
    """ Print mapped predicates first, then unmapped ones """
    global list_of_mapped_preds, analogyFilewriter
    global list_of_inferences
    # mapped, notMapped, unmapped, unmapped_target_preds_count = 0, 0, set(), 0
    t_pred_list, s_pred_list = [], []
    for t, s, u in list_of_mapped_preds:
        t_pred_list.append(t)
        s_pred_list.append(s)
        t_s, t_v, t_o = t
        s_s, s_v, s_o = s                # Full predicate mapped
        print("{: <20.20} {: >1} {: <10.10} {: >1} {: >20.20}".format(t_s, " ", t_v, " ", t_o), end="   ==   ")
        print("{: <20.20} {: >1} {: <10.10} {: >1} {: >20.20}".format(s_s, " ", s_v, " ", s_o), end="   ")
        rslt = wn_sim_mine(t_v, s_v, 'v')
        pred_sim_score = my_s2v(t_v, s_v, 'VERB') + my_s2v(t_s, s_s, 'NOUN') + my_s2v(t_o, s_o, 'NOUN')
        tmp = round((float(rslt[0]) + (float(rslt[2]))) / 2, 2)
        print(' {:>3.2f}'.format(pred_sim_score), " ", simplifyLCS(rslt[1]), simplifyLCS(rslt[3]))
        out_list = [t_s, t_v, t_o, "    ==    ", s_s, s_v, s_o, tmp]
        analogyFilewriter.writerow(out_list)
    return  # mapped, notMapped  # numeric summary


# ************************************************************************
# ************************* Cache and Similarity *************************
# ************************************************************************


def my_s2v_DEPRECATED(w1, w2, pos):
    if w1 == w2:
        return 1
    head_w1 = head_word(w1)
    head_w2 = head_word(w2)
    if pos == "VERB":
        if HOGS2.get_freq(head_w1 + '|VERB') is None or HOGS2.get_freq(head_w2 + '|VERB') is None:
            return 0
        else:
            return HOGS2.relational_distance(head_w1, head_w2)
    elif pos == "NOUN":
        freq1 = HOGS2.get_freq(head_w1 + '|NOUN')
        freq2 = HOGS2.get_freq(head_w2 + '|NOUN')
        if freq1 is None and freq2 is None:
            return 0.8 # novel words, probably open-class
        elif freq1 is None or freq2 is None:
            return 0  # only 1 novel openp-class word
        else:
            return HOGS2.conceptual_distance(head_w1, head_w2)


def my_s2v(w1, w2, pos):
    if pos == "VERB":
        return HOGS2.relational_distance(w1,w2)
    else:
        return HOGS2.conceptual_distance(w1, w2)


def read_wn_cache_to_dict():
    global CachePath
    global WN_cache
    WN_cache = {}
    if os.path.isfile(CachePath):
        with open(CachePath, "r") as wn_cache_file:
            filereader = csv.reader(wn_cache_file)
            for row in filereader:
                try:
                    WN_cache[row[0] + "-" + row[1]] = row[2:]
                except IndexError:
                    pass
read_wn_cache_to_dict()
print("WordNet Cache initialised.   ", end="")


def wn_sim_mine(w1, w2, partoS, use_lexname=True):
    """ wn_sim_mine("create","construct", 'v') -> [0.6139..., 'make(v.03)', 0.666..., 'make(v.03)'] """
    global LCSLlist
    if w1 == w2:
        return [1, w1, 1, w2]
    elif mode == 'Code':
        return [0, w1, 0, w2]
    lexname_out = ""
    lin_max, wup_max = 0, 0
    LCSL_temp = LCSW_temp = []
    flag = False
    w1 = w1.lower()
    w2 = w2.lower()
    wn1 = wn.morphy(w1, partoS)
    wn2 = wn.morphy(w2, partoS)
    if wn1 is not None:
        w1 = wn1
    if wn2 is not None:
        w2 = wn2
    LCSL = LCSW = "-"

    if w1 + "-" + w2 in WN_cache:
        zz = WN_cache[w1 + "-" + w2]
        if zz[0] == partoS:
            lin_max = zz[1]
            wup_max = zz[2]
            LCSL = zz[3]
            LCSW = zz[4]
    else:
        syns1 = wn.synsets(w1, pos=partoS)
        syns2 = wn.synsets(w2, pos=partoS)
        for ss1 in syns1:
            if use_lexname:
                ss1_lex_nm = ss1.lexname()
            for ss2 in syns2:
                lin = ss1.lin_similarity(ss2, brown_ic)  # semcor_ic) #brown_ic
                wup = ss1.wup_similarity(ss2)
                if lin > lin_max:
                    lin_\
                        = lin
                    ss1_Lin_temp = ss1
                    ss2_Lin_temp = ss2
                    LCSL_temp = ss1.lowest_common_hypernyms(ss2)
                if wup > wup_max:
                    wup_max = wup
                    ss1_Wup_temp = ss1
                    ss2_Wup_temp = ss2
                    LCSW_temp = ss1.lowest_common_hypernyms(ss2)  # may return []
                if lin is None:
                    lin = 0
                if wup is None:
                    wup = 0
                if use_lexname and flag ==  False:
                    if ss1_lex_nm == ss2.lexname() and flag == False:
                        lexname_out = ss1_lex_nm
                        flag = True
        if lin_max > 0:
            LCSL_temp = ss1_Lin_temp.lowest_common_hypernyms(ss2_Lin_temp)
            LCSL = simplifyLCSList(LCSL_temp)
        if wup_max > 0:
            LCSW_temp = ss1_Wup_temp.lowest_common_hypernyms(ss2_Wup_temp)
            LCSW = simplifyLCSList(LCSW_temp)
        if lin_max < 0.0000000001:
            lin_max = 0
            LCSW = simplifyLCSList(LCSW_temp)
        # print(" &&&& LCSW_temp2 ", LCSW_temp)
        if LCSW_temp == []:
            LCSW = "Synset('null." + partoS + ".02')"
        write_to_wn_cache_file(w1, w2, partoS, lin_max, wup_max, LCSL, LCSW+"-"+lexname_out)
    LCSLlist.append(LCSL)  # for the GUI presentation
    if LCSL_temp == []:
        LCSL = "Synset('null." + partoS + ".0404')"
    if LCSW_temp == []:
        LCSW = "Synset('null." + partoS + ".0403')"
    return [lin_max, LCSL, wup_max, LCSW+" "+lexname_out]
    # return [lin_max, LCSL, wup_max, LCSW, lexname_out]
# wn_sim("create","construct", 'v')


def write_to_wn_cache_file(w1, w2, pos, Lin, Wup, L_lcs, W_lcs):
    global WN_cache, CachePath
    if w1 == w2:
        return
    if bool(WN_cache) and w1+'-'+w2 in WN_cache:
        return
    else:
        with open(CachePath, "a+") as wn_cache_file:
            Stringtest = w1 + "," + w2 + "," + pos
            Stringtest += "," + str(Lin) + "," + str(Wup) + "," + L_lcs + "," + W_lcs   # + ","
            wn_cache_file.write(" \n" + Stringtest)


###########################################################################################################


def important_variables():
    """writeSummaryFileData(tgt_graph.graph['Graphid'], source_graph.graph['Graphid'],
        tgt_graph.number_of_predicates(), source_graph.number_of_predicates(),    #graph_edit_distance,
        number_of_mapped_predicates//len(best_analogy), number_connected_components, largest_connected_component,
        tgt_graph.number_of_nodes(), tgt_graph.number_of_edges(), source_graph.number_of_nodes(),
        source_graph.number_of_edges(), len(GM.mapping), averageS2V, count_s2v_one,
        avg_Lin_conceptual, avg_Wup_conceptual, avg_Lin_relational, avg_Wup_relational,
        number_of_inferences, mappedConcepts, average_relational_similarity,
        len(max_wcc), len(list_of_digraphs), GM.mapping['Total_Score'] ) """
    return 0

################################################################################################################

def evaluate_relational_distance(tRel, sRel):  # drive, walk
    """ #Lin0, #WuP0, #Lin1, #Wup1, LinSum, WuPSum,  {lexical super-category}} """
    global term_separator, mode
    if tRel == sRel:
        reslt = numpy.ones(7)
        reslt[0], reslt[1] = 0.0, 0.0    # reslt[2], reslt[3], reslt[4], reslt[5] = 1.0, 1.0, 1.0, 1.0
        LinLCS = tRel
        WuPLCS = tRel
    elif mode == 'Code':
        reslt = numpy.zeros(7)
        t_rel_split = tRel.split(term_separator)
        s_rel_split = sRel.split(term_separator)
        if t_rel_split[0] == s_rel_split[0]:   # Head identicality for relations?
            reslt[4] = 0.01
            reslt[5] = 0.01
            LinLCS = t_rel_split[0]
            WuPLCS = LinLCS
        elif (len(t_rel_split) > 1) and (t_rel_split[1] == s_rel_split[1]):  # 2nd word identicality
            reslt[4] = 0.21
            reslt[5] = 0.21
            LinLCS = t_rel_split[:1]
            WuPLCS = LinLCS
        else:
            reslt[4] = 1
            reslt[5] = 1
            LinLCS = 'no-reln'
            WuPLCS = 'no-reln'
    else:
        reslt = numpy.zeros(7)
        temp_result = wn_sim_mine(tRel, sRel, 'v')  # returns [0.6139, 'make(v.03)', 0.666, 'make(v.03)']
        if float(temp_result[0]) > 0 or float(temp_result[2]) > 0:
            if temp_result[0] == 0:
                reslt[0] = 1
            elif temp_result[0] == 1:
                reslt[2] = 1
            else:
                reslt[4] = temp_result[0]
            if temp_result[2] == 0:
                reslt[1] = 1
            elif temp_result[2] == 1:
                reslt[3] = 1
            else:
                reslt[5] = temp_result[2]
        LinLCS = temp_result[1]  # .find('(')
        WuPLCS = temp_result[3]
    return reslt, LinLCS, WuPLCS  # Lin0, WuP0, Lin1, Wup1, LinSum, WuPSum ...


def evaluate_conceptual_distance(tConc, sConc):  # ('cat','dog')
    global term_separator, mode
    reslt = numpy.zeros(7)  # Lin0, WuP0, Lin1, Wup1, LinSum, WuPSum,
    if tConc == sConc:
        reslt[2] = 0
        reslt[3] = 0
        reslt[4] = 0
        reslt[5] = 0
        LinLCS = tConc
        WuPLCS = tConc
    elif mode == 'Code':
        if head(tConc) == head(sConc):
            reslt[4] = 0
            reslt[5] = 0
            LinLCS = second_head(tConc)
            WuPLCS = LinLCS
        elif tConc.split(term_separator)[:1] == sConc.split(term_separator)[:1]:
            reslt[4] = 0.22
            reslt[5] = 0.22
            LinLCS = tConc.split(term_separator)[:1]
            WuPLCS = tConc.split(term_separator)[:1]
        elif tConc.split(term_separator)[0] == sConc.split(term_separator)[0]:
            reslt[4] = 0.33
            reslt[5] = 0.33
            LinLCS = tConc.split(term_separator)[0]
            WuPLCS = tConc.split(term_separator)[0]
        else:
            reslt[0] = 0.9991
            reslt[1] = 0.9991
            LinLCS = 'none'
            WuPLCS = 'none'
    else:
        temp_result = wn_sim_mine(tConc, sConc, 'n')  # returns [0.6139 'make(v.03)', 0.666, 'make(v.03)']
        # print("temp_result", temp_result)
        if float(temp_result[0]) > 0 or float(temp_result[2]) > 0:
            if temp_result[0] == 0:
                reslt[0] = 1
            elif temp_result[0] == 1:
                reslt[2] = 1
            else:
                reslt[4] == temp_result[0]
            if temp_result[2] == 0:
                reslt[1] = 1
            elif temp_result[2] == 1:
                reslt[3] = 1
            else:
                reslt[5] = temp_result[2]
        LinLCS = temp_result[1]
        WuPLCS = temp_result[3]
    return reslt, LinLCS, WuPLCS


########################################################################
########################################################################
########################################################################


def simplifyLCS(synsetName):
    """ Simplify a synsets to just the synset name.
    It accepts either an isolated synset name of a flat list of synsets.
    simplifyLCS(["Synset('object.n.01')"]) -> "object"""
    if isinstance(synsetName, list):
        synsetName = simplifyLCS(synsetName[0]) + simplifyLCS(synsetName[1:])
    elif (isinstance(synsetName, str)) and ("Synset" in synsetName):
        y = synsetName.find('(') + 2
        z = synsetName.find('.') + 5
        synsetName = synsetName[y:z].replace('.', '(', 1) + ")"
    elif (isinstance(synsetName, list)) and (len(synsetName) > 1):
        simplifyLCS(synsetName[0]).append(simplifyLCS(synsetName[1:]))
        simplifyLCS(str(synsetName[0])).append(simplifyLCSList(synsetName[1:]))  # 11/10
    elif str(synsetName)[:6] == "Synset":  # instance of <class 'nltk.corpus.reader.wordnet.Synset'>
        ssString = str(synsetName)
        y = ssString.find('(') + 2
        z = ssString.find('.') + 5
        synsetName = ssString[y:z].replace('.', '(', 1) + ")"
    return synsetName  # [synsetName]


# simplifyLCSList("[Synset('whole.n.02')]")

def simplifyLCSList(synsetList):
    if (synsetList is None):
        return "none1"
    elif (synsetList == []):
        return ""
    elif synsetList == "none":
        return "none"
    elif (isinstance(synsetList, str)):
        z = simplifyLCS(synsetList)
        return z
    elif (isinstance(synsetList, list)) and (len(synsetList) > 1):
        return str(simplifyLCS(synsetList[0])) + "_" + str(simplifyLCSList(synsetList[1:]))
    elif (isinstance(synsetList, list)) and (len(synsetList) == 1):
        return simplifyLCS(synsetList[0])
    else:
        print(" sLCSL5", end="")
        zz = simplifyLCS(synsetList)
        return zz


# #######################################################################
# ############################   Graph   ################################
# ############################  Matching   ##############################
# #######################################################################

def encode_graph_labels(grf):
    nu_grf = nx.DiGraph()
    s_encoding = {}  # label, number
    s_decoding = {}  # number, label
    label = 0
    for x, y in grf.edges():
        if x not in s_encoding.keys():
            s_decoding[label] = x
            s_encoding[x] = label
            label += 1
        if y not in s_encoding.keys():
            s_decoding[label] = y
            s_encoding[y] = label
            label += 1
        nu_grf.add_edge(s_encoding[x], s_encoding[y])
    return nu_grf, s_encoding, s_decoding


def find_most_similar_rel(t_reln, s_rel_list):
    rslt_list = []
    for z in s_rel_list:
        rslt_list.append(z + wn_sim_mine(t_reln, z, 'v'))
    best_match = sorted(rslt_list, key=lambda val: val[1], reverse=True)
    return best_match[0]


def mapping_process(target_graph, source_graph):
    global GM, relationMapping, list_of_mapped_preds, mapping_run_time, semantics
    global rel_s2v, rel_count, con_s2v, con_count
    rel_s2v, rel_count, con_s2v, con_count = 0, 0, 0, 0
    total_sim_score = 0
    if algorithm[0:4] == "HOGS":  # HOGS family
        if algorithm == "HOGS1":
            before_seconds = time.time()
            list_of_mapped_preds, number_mapped_predicates, mapping = \
                HOGS.generate_and_explore_mapping_space(target_graph, source_graph, semantics=semantics,
                                                        identical_edges_only=identical_edges_only)
            mapping_run_time = time.time() - before_seconds
        elif algorithm == "HOGS2":
            before_seconds = time.time()
            (list_of_mapped_preds, number_mapped_predicates, mapping, relatio_structural_dist,
             rel_s2v, rel_count, con_s2v, con_count) = HOGS2.generate_and_explore_mapping_space(target_graph, source_graph,
                                                                                                semantics=semantics,
                                                                     identical_edges_only=identical_edges_only)
            mapping_run_time = time.time() - before_seconds
            dud = 0
        print(" HOGS2 Time:", "{:.3f}".format(mapping_run_time), end="   ")
        print("   Rel Sim", rel_s2v, rel_count, "Con Sim",con_s2v,con_count)
        GM.mapping = {}
        for p, q, sim in list_of_mapped_preds:   # [0]:  # read back the results
            a, b, c = q
            x, y, z = p
            GM.mapping[a] = x
            GM.mapping[b] = y  # TODO: check for existing and append to multi-relation mappings
            GM.mapping[c] = z
            total_sim_score += sim
        GM.mapping['Total_Score'] = total_sim_score
        GM.mapping['Number_Mapped_Predicates'] = number_mapped_predicates
    elif algorithm == "VF2++":
        G = nx.Graph()
        G.add_edge(1, 2)
        # z1 = G.edges.data('label')
        for x, y in target_graph.edges():
            z = target_graph.edges.data('label')
            "label" in target_graph[x][y]
        res = nx.vf2pp_isomorphism(target_graph, source_graph)  # , node_label = "label")
    elif algorithm == "ismags":
        s_grf, s_encoding, s_decoding = encode_graph_labels(source_graph)
        t_grf, t_encoding, t_decoding = encode_graph_labels(target_graph)
        before_seconds = time.time()
        ismags = nx.isomorphism.ISMAGS(s_grf, t_grf)
        largest_common_subgraph = list(ismags.largest_common_subgraph(symmetry=False))  # False
        mapping_run_time = time.time() - before_seconds
        print(" ISMAGS Time:", mapping_run_time, end="  ")
        GM.mapping = largest_common_subgraph[0].copy()
        return largest_common_subgraph, s_decoding, t_encoding
    elif algorithm == "VF2":
        timeLimit = 30.0
        if __name__ == '__main__':
            # GM = isomorphvf2CB.MultiDiGraphMatcher(target_graph, source_graph)
            #p1 = multiprocessing.Process(target=isomorphvf2CB.MultiDiGraphMatcher,
            #         args=(target_graph, source_graph), name='MultiDiGraphMatcher')
            print(" VF2...", end="")
            p1.start()
            p1.join(timeout=timeLimit)
            p1.terminate()
        while p1.is_alive():
            print('.', end="")
            time.sleep(5)
        after_seconds = time.time()
        print("...VF2 ", after_seconds - before_seconds, end=" ")
        if p1.exitcode is None:     # a TimeOut
            return 0
        res = GM.subgraph_is_isomorphic()
        for s, v, o in returnEdgesAsList(target_graph): # choose matching edge
            if s in GM.mapping and o in GM.mapping:
                z = returnEdgesBetweenTheseObjects(GM.mapping[s], GM.mapping[o], source_graph)
                if len(z) == 0:
                    continue
                elif v in z:  # identical v
                    list_of_mapped_preds.append([[s, v, o], [GM.mapping[s], v, GM.mapping[o]]])
                elif len(z) == 1:
                    list_of_mapped_preds.append([[s, v, o], [GM.mapping[s], z[0], GM.mapping[o]]])
                else:
                    tmp = find_most_similar_rel(v, z)
                    list_of_mapped_preds.append([[s, v, o], [GM.mapping[s], tmp[0], GM.mapping[o]]])
        return GM.mapping
    if len(GM.mapping) == 0:
        print(":-( NO Mapping:")
    else:
        if target_graph.number_of_nodes() > 0:
            print("  ", list_diff(list(GM.mapping), ['Total_Score', 'Number_Mapped_Predicates']), " S-mapped words, ")
        else:
            print(" Empty target graph ")
    return []


def develop_analogy(target_graph, source_graph):
    global GM, relationMapping, semantics, generate_inferences
    global list_of_mapped_preds, mapping_graph, list_of_target_preds, num_novel_counterparts
    global num_identical_predicates, analogyFileName, max_wcc, max_wcc_nodes_list, list_of_inferences
    global mode
    global rel_s2v, rel_count, con_s2v, con_count
    analogyFileName = target_graph.graph['Graphid'] + "__"+source_graph.graph['Graphid'] + ".csv"
    num_identical_predicates, num_novel_counterparts = 0, 0
    max_wcc, max_wcc_nodes_list = 0, 0
    if target_graph.number_of_nodes() == 0 or source_graph.number_of_nodes() == 0:
        return [], 0  # FIXME need to return possibly 8 values
    if algorithm == "ismags":
        list_of_dictionaries, s_decoding, t_encoding = mapping_process(target_graph, source_graph, semantics)
        interpretations = []
        print("2 ", len(list_of_dictionaries), " interpretations.  ", end="")
        for dic in list_of_dictionaries:  # encode target but decode source
            res = addRelationsToMapping(target_graph, source_graph, dic, s_decoding, t_encoding)
            interpretations += [res]
        best_analogy = sorted(interpretations, key=lambda val: val[0], reverse=True)[0][1:]
        if best_analogy == []:
            list_of_inferences = []
        elif best_analogy[0] == 0:
            print("** Useless Mapping ** ", end="")
        else:
            for a, b, c, q, r, s in best_analogy[1:]:
                GM.mapping[a] = q
                GM.mapping[b] = r
                GM.mapping[c] = s
    else:
        best_analogy = mapping_process(target_graph, source_graph)  # HOGS
    if not algorithm == "ismags":   # isinstance(list_of_mapped_preds[0][0][0], list):
        best_analogy = slightly_flatten(list_of_mapped_preds)
    else:
        best_analogy = list_of_mapped_preds.copy()
    number_of_identical_predicates = 0
    if generate_inferences:
        generateCWSGInferences(target_graph, source_graph, best_analogy)

    mapping_graph = nx.MultiDiGraph()
    mapping_graph.clear()
    tot_semantic_distance, num_identical_predicates = 0, 0
    num_novel_counterparts = 0

    list_of_target_preds = returnEdgesAsList(target_graph)  # will contain unmapped target preds at the end
    #if mode == "code":
    #    list_of_mapped_preds = list_of_target_preds
    for pred1, pred2, scor in list_of_mapped_preds: # 10 June 2025 list_of_mapped_preds (slightly larger)
        if pred1[0] == pred2[0] and pred1[1] == pred2[1] and pred1[2] == pred2[2]:  # identical
            number_of_identical_predicates += 1
        mapping_graph.add_node(pred1[0]+"|"+pred2[0], label=pred1[0]+" | "+pred2[0])  # 4relist_of_mapped_preds = [[target-x, source-x] ...]
        mapping_graph.add_node(pred1[2]+"|"+pred2[2], label=pred1[2]+" | "+pred2[2])
        mapping_graph.add_edge(pred1[0]+"|"+pred2[0], pred1[2]+"|"+pred2[2], label=pred1[1]+" | "+pred2[1])
        if mode ==  "English":
            a = my_s2v(pred1[0], pred2[0], "NOUN")
            b = my_s2v(pred1[2], pred2[2], "NOUN")
            c = my_s2v(pred1[1], pred2[1], "VERB")
        elif mode == "Code":
            a = HOGS2.conceptual_distance(pred1[0], pred2[0])
            b = HOGS2.conceptual_distance(pred1[2], pred2[2])
            c = HOGS2.relational_distance(pred1[1], pred2[1])
        semantic_distance = a + b + c
        tot_semantic_distance += semantic_distance
        if semantic_distance == 0:
            num_identical_predicates += 1
        else:
            num_novel_counterparts += 1
        if [pred1[0], pred1[1], pred1[2]] in list_of_target_preds:
            list_of_target_preds.remove([pred1[0], pred1[1], pred1[2]])  # list_of_target_preds left with Unmapped preds
    if mapping_graph.number_of_edges() == 0:
        max_wcc = max_wcc_nodes_list = 0, []
    else:
        max_wcc_nodes_list = max(nx.weakly_connected_components(mapping_graph), key=len)

    if isinstance(max_wcc_nodes_list, set) and len(max_wcc_nodes_list) > 0:
        max_wcc = len(max_wcc_nodes_list)
    calculate_analogy_metrics(target_graph, source_graph, best_analogy)   # remove?
    if len(list_of_inferences) > 0:
        print("  ++INFERABLE:", len(list_of_inferences), list_of_inferences)
    return_analogy_result(target_graph, source_graph)
    return list_of_mapped_preds, tot_semantic_distance, (rel_s2v, rel_count, con_s2v, con_count)


def generateCWSGInferences(tgtGraph, srcGrf, mapped_preds_lis):  # generateCWSGInferences(sourceGraph, targetGraph)
    global GM
    global list_of_inferences
    global unmapped_target_edges
    list_of_inferences = []
    unmapped_target_edges = []
    if mapped_preds_lis == []:
        return
    src_edges = returnEdgesAsList(srcGrf)
    tgt_edges = returnEdgesAsList(tgtGraph)    # first remove mapped edges from input graphs
    for a, b, c, d, e, f in mapped_preds_lis:  # eliminate mapped predicates
        for h, i, j in src_edges:
            if a == h and b == i and c == j:
                src_edges.remove([h, i, j])
                break
        for h, i, j in tgt_edges:   # eliminate mapped predicates
            if d == h and e == i and f == j:
                tgt_edges.remove([h, i, j])
                break
    unmapped_target_edges = tgt_edges.copy()
    for subj, reln, obj in src_edges:   # unmapped source edges
        s = r = o = 0
        if subj in GM.mapping:
            s = 1
        if obj in GM.mapping:
            o = 1
        if reln in GM.mapping or reln in ['assert']:  # Aris - for code data only
            r = 1
        if reln in ['assert'] and mode == 'Code':
            print(" ASSERT found ", end="")
        subjMap = GM.mapping.get(subj)
        relnMap = GM.mapping.get(reln)
        objMap = GM.mapping.get(obj)
        if not predExists(subjMap, relnMap, objMap, tgtGraph):  # generate inference
            if s + r + o == 2:
                if subjMap is None:  # Generate transferable symbol
                    subjMap = subj   # + " ~INF"
                    relnMap = reln + " ~INF"
                if objMap is None:
                    objMap = obj #+ " ~INF"
                    relnMap = reln + " INF~"
                if relnMap is None:
                    relnMap = reln + " ~INF~"
                #print(" #INFER(",subj, ", ", reln, ", ", obj,")  =>  (",subjMap, ", ", relnMap, ", ", objMap, end=")")
                list_of_inferences = list_of_inferences + [[subjMap, relnMap, objMap, "FROM", subj, reln, obj]]
    print("\n", end="")



def calculate_analogy_metrics(tgt_graph, source_graph, best_analogy):  # source concept nodes
    """ Calculate Relational and Conceptual Similarity, where best_analogy = [[t0, t1, t2, s0, s1, s2], ...] """
    global analogyFilewriter, semcor_ic
    global mapping_graph, GM, list_of_mapped_preds
    global CSVPath, analogyFileName, CachePath, LCSLlist
    global mappedRelations, unmappedRelations, mappedConcepts, unmappedConcepts
    global list_of_inferences, edges_in_max_wcc, target_max_wcc_nodes
    global rel_s2v, rel_count, con_s2v, con_count
    mappedRelations, unmappedRelations, mappedConcepts, unmappedConcepts = 0, 0, 0, 0
    con_sim_vec = numpy.zeros(7)  # Lin0, WuP0, Lin1, Wup1, LinSum, WuPSum,
    rel_sim_vec = numpy.zeros(7)
    number_of_inferences = len(list_of_inferences)
    if not os.path.exists(os.path.dirname(CSVPath + analogyFileName)):
        try:
            os.makedirs(os.path.dirname(CSVPath + analogyFileName))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    unmappedRelations = tgt_graph.number_of_edges() - len(best_analogy)
    mappedRelations = len(best_analogy)

    with open(CSVPath + analogyFileName, 'w+') as analogyFile:   # ResultsOutput
        analogyFilewriter = csv.writer(analogyFile, delimiter=',',
                                       quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        analogyFilewriter.writerow(['Type', 'Word1', 'Word2', 'Lin', 'Wup', 'LCS Lin', 'LCS Wup'])
        analogyFilewriter.writerow([analogyFileName.partition("__")[0]])  # split-off target name
        analogyFilewriter.writerow([analogyFileName.partition("__")[2]])  # source name
        ############################
        # Relation Mapping Summary
        set_of_mapped_target_concepts = set()
        list_of_mapped_target_relations = []
        for t0, t1, t2, s0, s1, s2 in best_analogy:
            set_of_mapped_target_concepts.add(t0)
            list_of_mapped_target_relations.append(t1)
            set_of_mapped_target_concepts.add(t2)
            if mode == 'Code':
                res, Lin_LCS, Wu_LCS = evaluate_relational_distance(t1, s1)
                analogyFilewriter.writerow(['Rel', t1, s1, res[4], res[5], Lin_LCS, Wu_LCS])
                rel_sim_vec = rel_sim_vec + res
            elif mode == 'English':
                res, Lin_LCS, Wu_LCS = evaluate_relational_distance(t1, s1)
                analogyFilewriter.writerow(['Rel', t1, s1, res[4], res[5], Lin_LCS, Wu_LCS])
                rel_sim_vec = rel_sim_vec + res
        # ########################
        # Conceptual SIMILARITY
        #set_of_mapped_target_concepts = set()
        #list_of_mapped_target_relations = []  # yes, this one's a list
        #for x in tgt_preds:
        #    set_of_mapped_target_concepts.add(x[0])
        #    list_of_mapped_target_relations.append(x[1])
        #    set_of_mapped_target_concepts.add(x[2])
        for key in set_of_mapped_target_concepts :
            if key in GM.mapping:
                mappedConcepts += 1
                res, Lin_LCS, Wu_LCS =  evaluate_conceptual_distance(key, GM.mapping[key])
                analogyFilewriter.writerow(['Conn ', key, GM.mapping[key], res[4], res[5], Lin_LCS, Wu_LCS])
                con_sim_vec = con_sim_vec + res
        unmappedConcepts = len(list(tgt_graph.nodes())) - len(set_of_mapped_target_concepts)

        # Relations
        analogyFilewriter.writerow(['#Verb Lin==0', rel_sim_vec[0], 'of', (mappedRelations + unmappedRelations)])
        analogyFilewriter.writerow(['#Verb WuP==0', rel_sim_vec[1], 'of', (mappedRelations + unmappedRelations)])
        if mappedRelations + unmappedRelations == 0:
            avg_Lin_relational = 0
            avg_Wup_relational = 0
        else:
            avg_Lin_relational = rel_sim_vec[4] / (mappedRelations + unmappedRelations)
            avg_Wup_relational = rel_sim_vec[5] / (mappedRelations + unmappedRelations)
        analogyFilewriter.writerow(['Avg Verb Lin ', avg_Lin_relational])
        analogyFilewriter.writerow(['Avg Verb Wup ', avg_Wup_relational])
        analogyFilewriter.writerow(['#Verb Lin== 1', rel_sim_vec[2], 'of', (mappedRelations + unmappedRelations)])
        analogyFilewriter.writerow(['#Verb WuP== 1', rel_sim_vec[3], 'of', (mappedRelations + unmappedRelations)])
        # Concepts
        analogyFilewriter.writerow(['#Noun Lin==0', con_sim_vec[0], 'of', (mappedConcepts + unmappedConcepts)])
        analogyFilewriter.writerow(['#Noun WuP==0', con_sim_vec[1], 'of', (mappedConcepts + unmappedConcepts)])
        if mappedConcepts + unmappedConcepts == 0:
            avg_Lin_conceptual = 0
            avg_Wup_conceptual = 0
        else:
            avg_Lin_conceptual = con_sim_vec[4] / (mappedConcepts + unmappedConcepts)
            avg_Wup_conceptual = con_sim_vec[5] / (mappedConcepts + unmappedConcepts)
        analogyFilewriter.writerow(['Avg Noun Lin ', avg_Lin_conceptual])
        analogyFilewriter.writerow(['Avg Noun Wup ', avg_Wup_conceptual])
        analogyFilewriter.writerow(['#Noun Lin== 1', con_sim_vec[2], 'of', (mappedConcepts + unmappedConcepts)])
        analogyFilewriter.writerow(['#Noun WuP== 1', con_sim_vec[3], 'of', (mappedConcepts + unmappedConcepts)])

        average_relational_similarity = ((((avg_Lin_relational + avg_Wup_relational) / 2) * mappedRelations) * 0.5)
        #       +  ((((avgLin+avg_Wup_conceptual)/2)*mappedConcepts) * 0.5) )
        analogyFilewriter.writerow(['AnaSim=', average_relational_similarity])

        print(" ", tgt_graph.graph['Graphid'].rpartition(".")[0], source_graph.graph['Graphid'].rpartition(".")[0],
              " {:.2f}".format(avg_Lin_conceptual), " {:.2f}".format(avg_Wup_conceptual), con_sim_vec[2], " ",
              " {:.2f}".format(avg_Lin_relational), " {:.2f}".format(avg_Wup_relational),
              rel_sim_vec[2], " ", rel_sim_vec[3], " ", number_of_inferences, " ", mappedConcepts, " ", mappedRelations, " ",
              " {:.2f}".format(average_relational_similarity),
              GM.mapping['Number_Mapped_Predicates'], GM.mapping['Total_Score'] )

        edges_in_max_wcc = 0
        list_of_subgraphs = list(nx.weakly_connected_components(mapping_graph)) # unused
        list_of_digraphs = []
        for subgraph in list_of_subgraphs:
            list_of_digraphs.append(nx.subgraph(mapping_graph, subgraph))  # group nodes plus attached edges

        if mapping_graph.number_of_nodes() > 0:  # WCC - Weakly Connected Component
            mapping_max_wcc_nodes = max(nx.weakly_connected_components(mapping_graph), key=len)
        else:
            mapping_max_wcc_nodes = []
        max_wcc_subgraph = mapping_graph.subgraph(mapping_max_wcc_nodes)
        printMappedPredicates(tgt_graph)  # via list_of_mapped_preds

        # GROUNDED INFERENCES
        augment_mapping_graph_with_inferences_2() # list_of_inferences
        # for a, r, b in list_of_inferences:
        #    analogyFilewriter.writerow(["INFERENCE, ", a, r, b])
        #    mapping_graph.add_edge(a, b, label=r)
        # else:
        #    mapping_plus_inferences_wcc_nodes = set()
        if mapping_graph.number_of_nodes() != 0:
            mapping_plus_inferences_wcc_nodes = max(nx.weakly_connected_components(mapping_graph), key=len)
            infs_max_wcc_subgraph = len(mapping_graph.subgraph(mapping_plus_inferences_wcc_nodes).edges())
        else:
            mapping_plus_inferences_wcc_nodes, average_relational_similarity = 0, 0
            infs_max_wcc_subgraph = 0  # mapping_graph.subgraph(mapping_plus_inferences_wcc_nodes)
            if rel_count >= 1:
                average_relational_similarity = rel_s2v / rel_count
            if average_relational_similarity > 1:
                dud =0
        print(" dud set_of_mapped_target_concepts", set_of_mapped_target_concepts, "    ")
        writeSummaryFileData(tgt_graph.graph['Graphid'], source_graph.graph['Graphid'], tgt_graph.number_of_nodes(),
                tgt_graph.number_of_edges(), source_graph.number_of_nodes(), source_graph.number_of_edges(),
                len(best_analogy), len(set_of_mapped_target_concepts), len(max_wcc_subgraph.edges()),
                nx.number_weakly_connected_components(mapping_graph), avg_Wup_conceptual, avg_Wup_relational,
                avg_Lin_conceptual, avg_Lin_relational, average_relational_similarity, # via s2v
                number_of_inferences, len(mapping_max_wcc_nodes), infs_max_wcc_subgraph, # len(infs_max_wcc_subgraph.edges()),
                len(list_of_digraphs), GM.mapping['Total_Score'], algorithm, mapping_run_time,
                rel_s2v, rel_count, con_s2v, con_count)
        #TODO add; con_s2v, con_count
        print(len(best_analogy), len(set_of_mapped_target_concepts), len(max_wcc_subgraph.edges()),
                nx.number_weakly_connected_components(mapping_graph), avg_Wup_conceptual, avg_Wup_relational,
                avg_Lin_conceptual, avg_Lin_relational, average_relational_similarity,
                number_of_inferences, len(mapping_max_wcc_nodes), infs_max_wcc_subgraph, # len(infs_max_wcc_subgraph.edges()),
                len(list_of_digraphs), GM.mapping['Total_Score'], algorithm, mapping_run_time, con_s2v, con_count)
    # analogyFile.flush()
    # analogyFile.close()


def augment_mapping_graph_with_inferences_2():
    global GM
    global list_of_inferences
    global mapping_graph
    for a, r, b, _, x, p, y in list_of_inferences:
        a1 = get_reverse_mapping(a)
        b1 = get_reverse_mapping(b)
        if a1 in GM.mapping.keys() or b1 in GM.mapping.values(): # avoid cross-mapping
            print(" ---", a, r, b, end="    ")
            continue
        mapping_graph.add_node(a + "|" + x, label=a + "|" + x)
        mapping_graph.add_node(b + "|" + y, label=b + "|" + y)
        mapping_graph.add_edge(a + "|" + x, b + "|" + y, label=r)
        print(" INFR++(", a,r,b, end=")  ")


def get_reverse_mapping(x):
    global GM
    if x not in GM.mapping.values():
        return ""
    value = {i for i in GM.mapping if GM.mapping[i] == x}
    return list(value)[0]


def return_analogy_result(target_graph, source_graph):
    global list_of_mapped_preds
    global list_of_target_preds  # will eventually contain UN_mapped_target_preds
    global num_novel_counterparts
    global num_identical_predicates
    global mapping_graph
    global max_wcc
    global max_wcc_nodes_list
    global edges_in_max_wcc
    global target_max_wcc_nodes
    global unmapped_target_preds
    scor = 0
    unmapped_target_preds = list_of_target_preds.copy()
    if True and len(list_of_mapped_preds) > 0:   # for result visualisation only
        mapped_nodes = mapping_graph.nodes()
        for pred1 in list_of_target_preds:  # add UN-mapped target preds
            arg1 = pred1[0]
            arg2 = pred1[2]
            for z in mapped_nodes:
                graf_node_head_word = z.split("|")[0]
                if pred1[0] == graf_node_head_word:
                    arg1 = z
                if pred1[2] == graf_node_head_word:
                    arg2 = z
            mapping_graph.add_node(arg1, label=arg1)  # list_of_mapped_preds = [[target-x, source-x] ...]
            mapping_graph.add_node(arg2, label=arg2)
            mapping_graph.add_edge(arg1, arg2, label="**"+pred1[1])
            unmapped_target_preds.remove([pred1[0], pred1[1], pred1[2]])
        mapping_graph.graph['Graphid'] = target_graph.graph['Graphid'] + "__" + source_graph.graph['Graphid']
    return list_of_mapped_preds, scor, num_novel_counterparts, num_identical_predicates, \
        max_wcc_nodes_list, max_wcc, list_of_target_preds


def s_p_a_c_e():
    return


def addRelationsToMapping(target_graph, source_graph, mapping_dict, s_decoding, t_decoding):  # LCS_Number
    """new one. GM.mapping={(t,s), (t2,s2)...}"""
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
            # this_mapping[tNoun1_num] in this_mapping.values() and this_mapping[tNoun2_num] in this_mapping.values():
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
                    elif HOGS2.get_freq(head_word(tRelation) + '|VERB') is not None and \
                            HOGS2.get_freq(head_word(s_verb) + '|VERB') is not None:
                        simmy = HOGS2.similarity([head_word(tRelation) + '|VERB'], [head_word(s_verb) + '|VERB'])
                    # else:
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


#####################
# Open output files
#####################

# ("TARGET", "SOURCE", "#T Conc", "#T Rels", "GEdtDst", "#S Cons", "#S rels", "#Map Preds",  "#Map Conc",
# "AvLin Con", "AvWu Con", "AvLin Rel", "AvWu Rel", "Infs", "#MapCon", "MapRels", "AnaSim")

def writeSummaryFileData(*args):  # 21 params
    global CSVsummaryFileName
    global CSVPath
    CSVsummaryFileName = CSVPath + "summary.csv"
    out_list = []
    with open(CSVsummaryFileName, "a") as csvSummaryFileHandle:
        summaryFilewriter = csv.writer(csvSummaryFileHandle, delimiter=',',
                                        quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        for el in args:
            out_list.append(el)
        summaryFilewriter.writerow(out_list)

# ###############################################################
# ############### Run Multiple Analogies ########################
# ###############################################################

def resetAnalogyMetrics():
    global inference, list_of_inferences, list_of_mapped_preds
    global LCSLlist, GM
    global relationMapping
    inference = 0
    list_of_inferences = []
    LCSLlist = []
    list_of_mapped_preds = []
    # GM = isomorphvf2CB.MultiDiGraphMatcher(targetGraph, targetGraph)
    GM = MappingObject()
    GM.mapping['Total_Score'] = 0
    GM.mapping['Number_Mapped_Predicates'] = 0
    relationMapping = []


def blend_with_all_sources(targetFile):
    global all_csv_files, these_csv_files
    global target_graph, source_graph
    global analogyFileName
    global nextSourceFile
    global analogyCounter
    global max_graph_size
    global skip_over_previous_results
    global list_of_mapped_preds, unmapped_target_preds
    global mapping_graph, list_of_inferences, file_type_filter
    target_graph = build_graph_from_csv(targetFile).copy()
    target_graph.graph['Graphid'] = targetFile
    if target_graph.number_of_nodes() == 0:
        return
    p1 = targetFile.rfind(".")
    if target_graph.number_of_edges() > max_graph_size:
        target_graph = prune_peripheral_nodes(target_graph)
    for nextSourceFile in all_csv_files:
        if True and nextSourceFile == targetFile: # skip self comparison?
            continue
        p2 = nextSourceFile.rfind(".")
        print("\n\n#", mode, "================", "  ", targetFile[0:p1], "  <- ", nextSourceFile[0:p2], "======= ",
              end="")
        analogyFileName = targetFile + "__" + nextSourceFile + ".csv"
        if skip_over_previous_results and os.path.isfile(CSVPath + analogyFileName):
            continue
        resetAnalogyMetrics()
        temp_graph2.clear()
        source_graph = build_graph_from_csv(nextSourceFile).copy()
        source_graph.graph['Graphid'] = nextSourceFile
        if source_graph.number_of_edges() > max_graph_size:
            source_graph = prune_peripheral_nodes(source_graph)
        elif source_graph.number_of_nodes() == 0:
            continue
        develop_analogy(target_graph, source_graph)
        # develop_analogy(sourceGraph, targetGraph)
        if show_blended_graph:
            edg_lis = returnEdgesAsList(mapping_graph)
            list_of_inferences_shrt = [[a, b, c] for a, b, c, _, _, _, _ in list_of_inferences]
            ShowGraphs.show_blended_space_big_nodes(mapping_graph, edg_lis, [],
                                                    list_of_inferences_shrt, targetFile+"-"+nextSourceFile)
        analogyCounter += 1


def blend_to_possible_targets(sourceFile):
    global all_csv_files
    global target_graph
    global source_graph
    global analogyFileName
    global nextSourceFile
    global analogyCounter
    global max_graph_size
    print("\nExploring Source::", sourceFile, end="")
    analogyCounter = 0
    source_graph = build_graph_from_csv(sourceFile).copy()
    source_graph.graph['Graphid'] = sourceFile
    p1 = sourceFile.rfind(".")  # file_type_filter
    if source_graph.number_of_edges() > max_graph_size:
        source_graph = prune_peripheral_nodes(source_graph)
    for next_target_file in all_csv_files:
        if False and (next_target_file == sourceFile): # skip self comparison
           continue
        p2 = next_target_file.rfind(".")
        print("\n=====zzz=============", end=" ")
        print("#", mode, "   ", next_target_file[0:p2], "  <- ", sourceFile[0:p1], "=======")
        print("#==", mode, "   ", sourceFile, "  <- ", next_target_file[0:p2], "=======")
        analogyFileName = next_target_file[0:p2] + "__" + sourceFile[0:p1] + ".csv"
        resetAnalogyMetrics()
        temp_graph2.clear()
        target_graph = build_graph_from_csv(next_target_file).copy()
        target_graph.graph['Graphid'] = next_target_file
        predicate_based_summary(target_graph)
        if False and target_graph.number_of_edges() > max_graph_size:
            target_graph = prune_peripheral_nodes(target_graph)
        develop_analogy(target_graph, source_graph)
        print("Target map:", return_ratio_of_mapped_target_predicates(target_graph),
              "\tSource map:", return_ratio_of_mapped_target_predicates(source_graph))
        analogyCounter += 1


def blend_all_files():
    global all_csv_files
    global analogyCounter
    global CSVPath
    global these_csv_files
    global csvSumryFil
    global algorithm
    global file_type_filter
    analogyCounter = 0
    if not os.path.exists(CSVPath):
        os.makedirs(CSVPath)
    writeSummaryFileData("TARGET", "SOURCE", "#T Conc", "#T Rels", "#S Cons", "#S rels",
                         "#Map Preds",  "#Map Conc Unq", "Max WCC edges", "Num WCC", "AvWu Con",
                         "AvWu Rel", "AvLin Con",  "AvLin Rel", "Avg S2v Rel Sim", "#Infs", "Max WCC nodes", "Infs WCC nodes",
                         "# DiGraphs", "Tot Scor", "Algo", "Time", "Rel S2V", "Rel#", "ConS2v", "Con#")
    all_csv_files = sorted([i for i in all_csv_files]) # if i.endswith(".csv") and not "Aln." in i]) # and i.startswith(13 Trench")]
    these_csv_files = [i for i in all_csv_files if i.endswith(file_type_filter)] # and "T.txt" in i]
    # all_csv_files = [i for i in all_csv_files if i.endswith(file_type_filter) and i.startswith("1")]
    # for next_target_file in all_GPT35_files:
    for next_target_file in these_csv_files:
        print("\n\n------------------------------------------------------------------------------------------------")
        print("------------New Target: ", next_target_file, "------------------------------------------------------")
        print("--------------------------------------------------------------------------------------------------", end="")
        blend_with_all_sources(next_target_file)
        # blend_to_possible_targets(next_target_file) # treat "target" as a source
        stop()
    csvSumryFil.close()


def blend_with_select_sources(targetFile):
    global all_csv_files  # supply target, look for sources
    global target_graph
    global source_graph
    global analogyFileName
    global nextSourceFile
    global analogyCounter
    global max_graph_size
    global skip_over_previous_results, file_type_filter
    all_files = os.listdir(localPath)
    file_year = targetFile[0:2]
    locn1 = targetFile[3:].find(' ')
    file_author = targetFile[3:locn1]
    these_csv_files = [i for i in all_files if i.startswith(file_year) and not ".T." in i
                       and i.endswith(file_type_filter) and file_author in i]
    #all_GPT35_files = [i for i in these_csv_files if ".GPT35.S." in i and i.endswith(".csv")]
    #all_GPT35_files.sort(reverse=True)
    print(" ", len(these_csv_files), " candidate sources. ")
    target_graph = build_graph_from_csv(targetFile).copy()
    target_graph.graph['Graphid'] = targetFile
    p1 = targetFile.rfind(".")
    if target_graph.number_of_edges() > max_graph_size:
        target_graph = prune_peripheral_nodes(target_graph)
    if show_input_graph:
        edg_lis = returnEdgesAsList(target_graph)
        ShowGraphs.show_blended_space_big_nodes(target_graph, edg_lis, [], [],
                                                output_filename=target_graph.graph['Graphid'])
    for nextSourceFile in these_csv_files:
        if True and nextSourceFile == targetFile:  # skip self comparison?
            continue
        p2 = nextSourceFile.rfind(".")
        print("\n\n#", mode, "================", "  ", targetFile, "  <- ", nextSourceFile[0:p2], "=======")
        analogyFileName = targetFile[0:p1] + "__" + nextSourceFile[0:p2] + ".csv"
        # analogyFileName = targetFile + "__" + nextSourceFile + ".csv"
        if skip_over_previous_results and path.isfile(CSVPath + analogyFileName):
            print(" £skippy ", end="")
            continue
        resetAnalogyMetrics()
        temp_graph2.clear()
        source_graph = build_graph_from_csv(nextSourceFile).copy()
        source_graph.graph['Graphid'] = nextSourceFile
        if show_blended_graph:
            edg_lis = returnEdgesAsList(source_graph)
            ShowGraphs.show_blended_space_big_nodes(source_graph, edg_lis, [], [],
                                                    output_filename=target_graph.graph['Graphid'])
        if source_graph.number_of_edges() > max_graph_size:
            source_graph = prune_peripheral_nodes(source_graph)
        develop_analogy(target_graph, source_graph)  # develop_analogy(sourceGraph, targetGraph)
        print("Source map:", return_ratio_of_mapped_source_predicates(source_graph),  # ...Source_Predicates
              "\tTarget map:", return_ratio_of_mapped_target_predicates(target_graph))
        analogyCounter += 1
        if show_blended_graph:
            edg_lis = returnEdgesAsList(mapping_graph)
            list_of_inferences_shrt = [ [a,b,c] for a,b,c,_,_,_,_ in list_of_inferences]
            ShowGraphs.show_blended_space_big_nodes(mapping_graph, edg_lis, [],
                                                    list_of_inferences_shrt, targetFile + "-" + nextSourceFile)
        # analogyFileName.close()


def blend_with_non_matching_sources(targetFile):
    import random
    global all_csv_files  # supply target, look for sources
    global target_graph, source_graph
    global analogyFileName
    global nextSourceFile
    global analogyCounter
    global max_graph_size
    global skip_over_previous_results
    if False:
        substring_location = targetFile.find('task')
        task = targetFile[substring_location:substring_location+5]
        these_csv_files = [i for i in all_csv_files if i[substring_location:substring_location+5] == task]
    all_files = os.listdir(localPath)
    ### for the Out Group comparisons
    position_of_header = targetFile.find(" ", targetFile.find(" ") + 1)  # 2nd occurrence of space
    these_csv_files = [i for i in all_files if i.endswith(file_type_filter) and not i.startswith(targetFile[0:position_of_header])]
    #all_GPT35_files = [i for i in these_csv_files if ".GPT35.S." in i and i.endswith(".csv")]
    #all_GPT35_files.sort(reverse=True)
    print(len(these_csv_files), " candidate sources.")
    target_graph = build_graph_from_csv(targetFile).copy()
    target_graph.graph['Graphid'] = targetFile
    p1 = targetFile.rfind(".")  # file_type_filter
    # print("SUMRY: ", predicate_based_summary(targetGraph), end=" ")
    if target_graph.number_of_edges() > max_graph_size:
        target_graph = prune_peripheral_nodes(target_graph)
    if show_input_graph:
        edg_lis = returnEdgesAsList(target_graph)
        ShowGraphs.show_blended_space_big_nodes(target_graph, edg_lis, [], [],
                                                output_filename=target_graph.graph['Graphid'])
    random.shuffle(these_csv_files)
    these_csv_files = these_csv_files[0:10]
    for nextSourceFile in these_csv_files:
        if True and nextSourceFile == targetFile:  # skip self comparison
            continue
        p2 = nextSourceFile.rfind(".")
        print("\n\n#", mode, "================", "  ", targetFile, "  <- ", nextSourceFile[0:p2], "=======")
        analogyFileName = targetFile[0:p1] + "__" + nextSourceFile[0:p2] + ".csv"
        if skip_over_previous_results and path.isfile(CSVPath + analogyFileName):
            print(" £skippy ", end="")
            continue
        resetAnalogyMetrics()
        temp_graph2.clear()
        source_graph = build_graph_from_csv(nextSourceFile).copy()
        source_graph.graph['Graphid'] = nextSourceFile
        if show_blended_graph:
            edg_lis = returnEdgesAsList(source_graph)
            ShowGraphs.show_blended_space_big_nodes(source_graph, edg_lis, [], [],
                                                    output_filename=target_graph.graph['Graphid'])
        if source_graph.number_of_edges() > max_graph_size:
            source_graph = prune_peripheral_nodes(source_graph)
        develop_analogy(target_graph, source_graph)  # develop_analogy(sourceGraph, targetGraph)
        print("Source map:", return_ratio_of_mapped_source_predicates(source_graph),  # ...Source_Predicates
              "\tTarget map:", return_ratio_of_mapped_target_predicates(target_graph))
        analogyCounter += 1
        if show_blended_graph:
            edg_lis = returnEdgesAsList(mapping_graph)
            list_of_inferences_shrt = [ [a,b,c] for a,b,c,_,_,_,_ in list_of_inferences]
            ShowGraphs.show_blended_space_big_nodes(mapping_graph, edg_lis, [],
                                                    list_of_inferences_shrt, targetFile + "-" + nextSourceFile)
        # analogyFileName.close()



def blend_file_groups():  # iterate over all targets and their related sources
    global all_csv_files
    global analogyCounter
    global CSVPath
    global GM
    global csvSumryFil
    global algorithm, file_type_filter
    analogyCounter = 0
    if not os.path.exists(CSVPath):
        os.makedirs(CSVPath)
    writeSummaryFileData("TARGET", "SOURCE", "#T Conc", "#T Rels", "#S Cons", "#S rels", "#Map Preds",  "#Unq Map Conc",
                         "Max WCC edges", "# WCCs", "AvWu Con", "AvWu Rel", "AvLin Con",  "AvLin Rel",  "Avg Rel Sim",
                         "#Infs", "Max WCC nodes", "Infs WCC nodes", "# DiGraphs", "Tot Scor", "Algo", "Time",
                         "Rel S2V", "Rel#", "ConS2v", "Con#")
    sourceFiles = os.listdir(localPath)
    these_csv_files = [i for i in sourceFiles if i.endswith(file_type_filter) and ".T." in i]  # and "98 Ratterman E3" in i]
    these_csv_files.sort()
    other_csv_files = [i for i in all_csv_files if ".T." in i]  # .endswith(".T.OpIE")]
    print("SEVERE RESTRICTION - Mapping File GROUPS only. # Files=", end="")
    print(len(these_csv_files))
    for next_target_file in these_csv_files:  # these_csv_files:
        print("\n\n-----New Target: ", next_target_file, "------------------------------------------------------\n")
        blend_with_select_sources(next_target_file)
        # blend_with_non_matching_sources(next_target_file)
        analogyCounter += 1
        print("\n\n")


def threeWordSummary(Grf):
    simplifiedGraf = nx.Graph(Grf)  # simplify to NON-Multi - Graph
    from networkx.algorithms import tree
    mst = tree.minimum_spanning_edges(simplifiedGraf, algorithm="kruskal", data=False)
    edgelist = list(mst)
    pr = nx.pagerank(simplifiedGraf, alpha=0.8)
    predsList = returnEdgesAsList(Grf)
    predsList2 = []
    for (n1, n2) in edgelist:
        for [n3, r, n4] in predsList:
            if (n1 == n3 and n2 == n4) or (n1 == n4 and n2 == n3):
                scor = pr[n1] * pr[n2]
                predsList2.append([scor, n1, r, n2])
                break
    predsList2.sort(key=lambda x: x[0], reverse=True)
    print()
    for [_, n1, r, n2] in predsList2[0:5]:
        print(n1, r, n2, end="   ")
    print()


def prune_peripheral_nodes(grf):
    global max_graph_size
    print("Pruning nodes from ", grf.number_of_edges(), end="-> ")
    affected_nodes = set()
    while grf.number_of_edges() > max_graph_size:
        degr_list = []
        for (a, b, r) in grf.edges.data('label'):
            degr_list += [[grf.degree(a)+grf.degree(b), a, b, r]]
        degree_sequence = sorted(degr_list, key=lambda val: val[0])
        for num, a, b, r in degree_sequence:
            grf = delete_one_edge(grf, a, b)
            affected_nodes.add(a)
            affected_nodes.add(b)
            if grf.number_of_edges() <= max_graph_size:
                break
    grf.remove_nodes_from(list(nx.isolates(grf)))   # remove edge-less nodes
    print(grf.number_of_edges())
    return grf


def delete_one_edge(grf, a, b):
    for (u, v, rel) in grf.edges.data('label'):
        if a == u and b == v:
            e = (u, v, {"label": rel})
            grf.has_edge(*e[:2])
            grf.remove_edge(*e[:2])
            break
    return grf

def predicate_based_summary(grf):
    edge_list = returnEdgesAsList(grf)
    result = []
    pr = nx.pagerank_numpy(grf)   # nx.pagerank(grf)  networkx3.0
    for s, v, o in edge_list:
        scor = grf.in_degree(s) + grf.out_degree(s) + grf.in_degree(o) + grf.out_degree(o)
        scor2 = pr[s] + pr[o]
        result.append([scor, scor2, s, v, o])
    best_analogy = sorted(result, key=lambda val: val[0], reverse=True)
    # centroid of coincident relations
    if len(best_analogy) >0:
        return best_analogy[0]
    else:
        return []


# #############################################################################
# ##########################   ConceptNet    ##################################
# #############################################################################


# CN Elaboration, CN_file -> CN_dict
def loadConceptNet(CNfileNam):
    """ Load ConceptNet data for graph elaboration"""
    global CN_dict
    # csvreader = csv.reader(CNfileNam, delimiter=',', quotechar='|')
    reader = csv.reader(open(CNfileNam, 'r'))  # , errors='replace')
    for row in reader:
        noun1, noun2, verb = row
        try:
            CN_dict[noun1].append((verb, noun2))
        except KeyError:
            CN_dict[noun1] = [(verb, noun2)]
loadConceptNet(CN_file_name)


def checkCNConnection(noun1, noun2):
    # Returns true if CN_dictionary has noun1 as key & value (any verb, noun2)
    r = False
    try:
        for pair in CN_dict[noun1]:
            if not r:
                r = noun2 in pair[1]
    except KeyError:
        r = False
    return r


global CN_relations_dict
CN_relations_dict = {'FormOf': 'Form_Of', 'PartOf': 'Part_Of', 'AtLocation': 'At_Location', 'HasA': 'Has_A',
    'HasProperty': 'Has_property', 'MadeOf':'Made_Of', 'IsA':'Is_a', 'HasContext': 'Has_Context',
    'CapableOf': 'Capable_Of', 'CausesDesire': 'Causes_Desire', 'HasPrerequisite': 'Has_Prerequisite',
    'HasLastSubevent': 'Has_Last_Subevent', 'HasFirstSubevent': 'Has_first_Subevent',
    'IsInstanceOf': 'Is_Instance_Of', 'MotivatedByGoal': 'Motivated_By_Goal',
    'NotDesires': 'Not_desires', 'NotHasProperty': 'Not_Has_Property', 'ReceivesAction': 'Receives_Action',
    'UsedFor': 'Used_For'}
def rewrite_CN_relation(rel):
    global CN_relations_dict
    if rel in CN_relations_dict.keys():
        return CN_relations_dict[rel]
    else:
        return rel

def getCNConnections_BACKUP(noun1, noun2):
    """ Return the first (single) connecting edge between graph nodes, bi-directional """
    global CN_dict
    connection = []
    if noun1 in CN_dict:
        for pair in CN_dict[noun1]:
            if noun2 == pair[1]:
                connection.append(pair[0])
                break
    if connection == [] and noun2 in CN_dict:
        for pair in CN_dict[noun2]:
            if noun1 == pair[1]:
                connection.append(pair[0])
                break
    #if connection != []:
    #    print(" CN ", noun1, connection, noun2, end=" ")
    return connection

def getCNConnections(noun1, noun2):
    """ Return the first (single) connecting edge between graph nodes, directionally sensitive """
    global CN_dict
    connection = []
    if noun1 in CN_dict.keys():
        for pair in CN_dict[noun1]:
            if noun2 == pair[1]:
                connection.append([noun1, pair[0], noun2])
                break
    if connection == [] and noun2 in CN_dict.keys():
        for pair in CN_dict[noun2]:
            if noun1 == pair[1]:
                connection.append([noun2, rewrite_CN_relation(pair[0]), noun1])
                break
    return connection
# getCNConnections('four','4')
# getCNConnections("twin", "twins")
# getCNConnections('road', 'country')
# getCNConnections('hawk', 'feathers')
# getCNConnections('feathers', 'hawk')



def findMissingConnections(graph):
    """ link nodes with common lemma XOR likely ConceptNet relations. """
    global term_separator
    global CN_dict
    nodes = list(graph.nodes)
    proposed_connections = {}
    for indexa in range(len(nodes)):
        head_a = nodes[indexa].split(term_separator)[0]
        a = wnl.lemmatize(head_a)
        for indexb in range(indexa + 1, len(nodes)):
            if graph.has_edge(nodes[indexa], nodes[indexb]) or \
                    graph.has_edge(nodes[indexb], nodes[indexa]):  # minimal disturbance principle
                break
            head_b = nodes[indexb].split(term_separator)[0]
            if head_a != head_b:
                b = wnl.lemmatize(head_b)
                if a in proposed_connections and proposed_connections[a] == b:
                    break
                elif b in proposed_connections and proposed_connections[b] == a:
                    break
                if a == b:  # country ~ countries
                    print("** WN LEMMA ", head_a, head_b, end="  ")
                    graph.add_edge(nodes[indexa], nodes[indexb], label="Form_Of *")
                    proposed_connections[a] = b
                    break
                else:
                    rel = getCNConnections(a, b)  # directionally sensitive
                    print(" REL", rel, end=" ")
                    if len(rel) > 0:
                        graph.add_edge(rel[0], rel[2], label=rel[1]) # rel[0]
                        proposed_connections[a] = b
                        print(" ** CN:", rel[0], rel[1], rel[2], end=" ")
                    elif rel != []:
                        graph.add_edge(a, b, label=rel) # rel[0]
                        proposed_connections[a] = b
                        print(" ** WN:", a, rel, b, end=" ")
                        print(" ** WN:", rel[0], rel[1], rel[2], end=" ")
    print()


def addConnection(noun1, noun2, verb):
    # Adds the connection noun1 -> verb -> noun2
    if noun1 in Graph_dictionary:
        if (verb, noun2) not in Graph_dictionary[noun1]:
            Graph_dictionary[noun1].append((verb, noun2))
            print("added-a " + noun1 + " " + verb + " " + noun2)
    else:
        Graph_dictionary[noun1] = [(verb, noun2)]
        print("added-b " + noun1 + " " + verb + " " + noun2)
        # Prints the keys/values of the newly added connection


def findNewNodes(graph):
    # Creates a list containing all the current nodes in our graph
    #nodes = getNodes(graph)
    nodes = graph.nodes()
    # Iterates over groups of three nodes and checks if they share a common connected node using shareCommonNode
    # Change this to four nodes if you wish to increase the requirement
    for n1 in nodes:
        for n2 in nodes:
            for n3 in nodes:
                if n1 != n2 and n2 != n3 and n1 != n3:
                    toAdd0 = shareCommonNode(n1, n2, n3)
                    if toAdd0 != []:
                        # If all three nodes share a common connected node
                        # That node is added along with the relevant connection
                        for node in toAdd0:
                            for verb in getCNConnections(n1, node):
                                if verb not in BannedEdges:
                                    addConnection(n1, node, verb)
                            for verb in getCNConnections(n2, node):
                                if verb not in BannedEdges:
                                    addConnection(n2, node, verb)
                            for verb in getCNConnections(n3, node):
                                if verb not in BannedEdges:
                                    addConnection(n3, node, verb)
    print('')
    for key in CN_dictionary:
        toAdd1 = []
        for pair in CN_dictionary[key]:
            if pair[1] in nodes:
                toAdd1.append(pair[1])
        if len(toAdd1) >= 3:
            # This enforces the restriction of three current nodes, edit if you wish to change the requirement
            for node in toAdd1:
                for verb in getCNConnections(key, node):
                    if verb not in BannedEdges:
                        addConnection(key, node, verb)



#######################################################################################################


def quick_wasserstein(a, b):  # inspired by Wasserstein distance (quick Wasserstein approximation)
    a_prime = sorted(list(set(a) - set(b)))
    b_prime = sorted(list(set(b) - set(a)))
    if len(a_prime) < len(b_prime):  # longer list is the prime
        temp = b_prime.copy()
        b_prime = a_prime.copy()
        a_prime = temp.copy()
    sum1 = sum(abs(i - j) for i, j in zip(a_prime, b_prime))
    b_len = len(b_prime)
    sum2 = sum(a_prime[b_len:])
    return sum1 + sum2
assert quick_wasserstein([1, 2, 3], [1, 2, 3]) == 0
assert quick_wasserstein([1, 2, 3], [1, 2, 4, 5]) == 6
assert quick_wasserstein([0, 1, 3], [5, 6, 8]) == 15
# assert quick_wasserstein([1, 2, 2, 1, 1], [2, 2, 2, 1, 1]) == 1


loadConceptNet(CN_file_name)  # CN_dict
print("ConceptNet loaded", end=" ")
getCNConnections('four', '4')

resetAnalogyMetrics()


def retrun_psych_material_head(fil_nam):  # From Words and Topologies.py. please delete here
    locn1 = fil_nam[4:].find(" ")
    locn2 = fil_nam[4:].find("-")
    if locn1 == -1:
        locn2 = 9999
    if locn2 == -1:
        locn2 = 9999
    locn = min(locn1, locn2)
    head_name = fil_nam[0:locn + 4]
    return head_name


def generate_all_graph_metrics():
    from statistics import mean
    global perform_eager_concept_fusion, term_separator
    term_separator = "_"
    node_parts, edge_parts = [], []
    perform_eager_concept_fusion = False
    all_files = os.listdir(localPath)
    these_csv_files = [i for i in all_files if i.endswith(".Aln")]# and not "GPT35" in i]  # or i.endswith(".Aln")
    #                   or i.endswith(".csv") or i.endswith(".OpIE")]
    these_csv_files.sort()
    if 'Psych Story Texts Metrics.csv' in these_csv_files:
        these_csv_files.remove('Psych Story Texts Metrics.csv')
    if 'Cache.txt' in these_csv_files:
        these_csv_files.remove('Cache.txt')
    with open(base_path + "/21 Karp NP" + '/IE-graph metrics num-Nodes num-Edges.txt', 'a') as fyle:
        for fil in these_csv_files:
            print(fil, end="   ")
            targetGraph = build_graph_from_csv(fil)
            targetGraph.graph['Graphid'] = targetGraph.graph['Graphid'] + "YsFuse"
            num_nodes = targetGraph.number_of_nodes()
            num_edges = targetGraph.number_of_edges()
            num_conn_components = len(list(nx.weakly_connected_components(targetGraph)))
            list_cc = list(nx.weakly_connected_components(targetGraph))
            biggest_cc = 0
            for item in list_cc:
                if len(item) > biggest_cc:
                    biggest_cc = len(item)
            max_degree = 0
            degrees = dict(targetGraph.degree())
            if degrees == {}:
                max_degree, in_arity, out_arity = 0, 0, 0
            else:
                max_degree = max(degrees.values())
                in_arity = sorted(d for n, d in targetGraph.in_degree())[-1]
                out_arity = sorted(d for n, d in targetGraph.out_degree())[-1]
            words_per_node, words_per_edge = 0, 0
            term_separator = "_"
            for n in targetGraph.nodes():
                cnt = n.count(term_separator)
                words_per_node += cnt
                node_parts.append(cnt)
            term_separator = '_'
            for n in targetGraph.edges(data=True):
                relation = n[2]['label']
                cnt = relation.count(term_separator)
                words_per_edge += cnt
                edge_parts.append(cnt)
            mean_node_parts, mean_edge_parts = 0, 0
            if not len(node_parts) == 0:
                mean_node_parts = mean(node_parts)
            if not len(edge_parts) == 0:
                mean_edge_parts = mean(edge_parts)
            publictn = retrun_psych_material_head(fil)
            lizt = [str(fil).replace(",",""), "#N, #E, #ConnectComponent, #WeakConnectComponen, Max_Degree, in_Deg, Out_Deg",
                    num_nodes, num_edges, num_conn_components, biggest_cc, max_degree, in_arity, out_arity]
            #lizt = [str(fil), publictn, "N, E, W/n, W/e, AvgW/n, AvgW/e", str(num_nodes), str(num_edges),
            #        num_conn_components, largest_cc, str(arity[0:4]), 'dog',
            #        str(words_per_node), str(words_per_edge), str(mean_node_parts), str(mean_edge_parts)]
            out_put = str(lizt) + "\n"
            fyle.write(out_put)
    stop()
# generate_all_graph_metrics()


def generate_all_FDG_graphs():  # FDG - Force Directed Graph
    import ShowGraphs
    global perform_eager_concept_fusion
    global skip_over_previous_results
    perform_eager_concept_fusion = False
    localPath = 'C:/Users/dodonoghue/Documents/Python-Me/data/Aesops Fables'
    source_files = os.listdir(localPath)  # py-Csvs
    these_csv_files = [i for i in source_files if  i.endswith('.RVB')] # and not i.endswith('.txt')]# and i.startswith("15 Murphy")]
    if not os.path.exists(os.path.dirname(localPath + "FDG/")):  # skip
        os.makedirs(localPath + "FDG/")
    # fdg_file = 'C:/Users/dodonoghue/Documents/Python-Me/Cre8blend/FDG/'
    fdg_file = 'C:/Users/dodonoghue/Documents/Python-Me/data/Aesops Fables/FDG'
    for fil in these_csv_files:
        if True and os.path.isfile(fdg_file+"FDG/" + fil+".html"):  # skip over
            print("skip", end="")
            continue
        print(" ",fil, end="   ")
        targetGraph = build_graph_from_csv(fil)
        targetGraph.graph['Graphid'] = targetGraph.graph['Graphid'] #+ "YsFuse"
        edg_lis = returnEdgesAsList(targetGraph)
        if True:
            ShowGraphs.show_blended_space_big_nodes(targetGraph, edg_lis, [], [], output_filename=targetGraph.graph['Graphid'])
    path = localPath + "FDG/"
    #print("FDG in ", fdg_file )
    os.startfile(fdg_file)  # display the directory contents
    stop()
# generate_all_FDG_graphs()

blend_all_files()
# stop()
# blend_file_groups()
