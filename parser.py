# py2 

import re
import os
import sys
import math
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--run_in_shell', default="False", help='run the parser in the shell...')
parser.add_argument('--test', default="True", help='test with --head lines in the corpus')
parser.add_argument('--head', default=500)
parser.add_argument('--train_p', type=float, default=0.8)
parser.add_argument('--valid_p', type=float, default=0.1)
parser.add_argument('--test_p', type=float, default=0.1)
parser.add_argument('--remove_hyphen', default="True")
parser.add_argument('--preprocessed_file', default='./preprocessed_file.txt')
parser.add_argument('--output_dir', default='./output.txt')
parser.add_argument('--test_dir', default='./test.txt')
parser.add_argument('--train_dir', default='./sequoia-corpus+fct.mrg_strict')
parser.add_argument('--tagger_model_dir', type=str, default='stanford-postagger-full-2017-06-09/models/french.tagger')
parser.add_argument('--jar_model_dir', type=str, default='stanford-postagger-full-2017-06-09/stanford-postagger.jar')

args = parser.parse_args()

from nltk import Tree
from nltk import PCFG
from nltk.tag import StanfordPOSTagger
from nltk import word_tokenize
from nltk import Production, nonterminals

# set environment variables for the pos french lexicon
jar = args.jar_model_dir
model = args.tagger_model_dir
pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8')

# add a method for checking unknown word in the grammar
def check_unknown_words(tokens, grammar):
    unknownlist = []
    for i, token in enumerate(tokens):
        try:
            grammar.check_coverage([token])
        except ValueError:
            unknownlist.append(token)
            
    return unknownlist
    # try:
    #     grammar.check_coverage(tokens)
    #     return None
    # except ValueError, e:
    #     print e
        #l = e.message
        #temps_ = re.findall(": .(.+).", l)[0].replace("\'", '').replace('\"', '').replace('\\\\', '\\').split(', ')
        #return list(map(lambda s:s[1:], temps_)) # remove the u at 0 position for each word
    
def update_grammar(productions, unknown):
    lis = pos_tagger.tag(unknown)
    for i in range(len(lis)):
        pos = nonterminals(lis[i][1])[0]
        production_ = Production(pos, [unknown[i]])
        productions.append(production_)
        print production_, "added to productions"

    S = Nonterminal('SENT')
    grammar = induce_pcfg(S, productions)
    
    return grammar


# largest lines of the file 
LIMIT = 5000
# the file for train and the preprocessed one
FILE = args.train_dir
processed_file = args.preprocessed_file
# the train data should be separate into

#  0.8 0.1 0.1 portion by default
train_p = args.train_p
valid_p = args.valid_p
test_p = args.test_p

######## remove the hypen and preprocess ############

hypen_exp = '(\([a-zA-Z]+)[\-\+][a-zA-Z_]+(\s)'

if args.remove_hyphen == "True":
    if not os.path.exists(processed_file):
        g = open(processed_file, 'wb')
        with open(FILE, 'rb') as f:
            for line in f.readlines():
                line = "".join(re.split(hypen_exp, line.decode('utf-8').strip()))
                g.write((line + "\n").encode('utf-8'))
        g.close()
    FILE = processed_file

if args.test == "True":
    LIMIT = args.head

########### start processing : read treebank ##########

treebank = []
productions = []
# use the index to split the data set
train_idx, valid_idx, test_idx = 0, 0, 0

with open(FILE, 'rb') as f:
    idx = 0
    alllines = f.readlines()
    # split the data set, but respect the limit
    length = min(len(alllines), LIMIT)
    train_idx = int(math.floor(length * train_p))
    valid_idx = int(math.floor(length * (train_p + valid_p)))
    test_idx = length-1

    for line in alllines:
        if idx >= LIMIT:
            break

        tree = Tree.fromstring(line.decode('utf-8').strip())
        treebank.append(tree.copy())

        if idx <= train_idx:
            tree.chomsky_normal_form(horzMarkov = 2) # Remove A->(B,C,D) into A->B,C+D->D
            productions += tree.productions()
            print tree  # after the chomsky normalisation
        idx += 1

# test the tree 
# for i in range(len(treebank)):
#     print treebank[i]
#print productions

############ create PCFG from the productions #######
from nltk import Nonterminal
from nltk import induce_pcfg

S = Nonterminal('SENT')
grammar = induce_pcfg(S, productions)
print(grammar)


######### Parser with CYK dynamic algorithm ########
from nltk.parse import pchart
from nltk.parse import ViterbiParser
from nltk.treetransforms import un_chomsky_normal_form

parser = ViterbiParser(grammar) 
parser.trace(2)

parses_bank = []

test_file = open(args.test_dir, 'wb')
test_output_file = open(args.output_dir, 'wb')

for i in range(valid_idx, test_idx+1):
    # take the leaves of each tree of testset and store
    # them in the test file 
    tokens = treebank[i][0].leaves()
    sentence = u" ".join(tokens)
    test_file.write((sentence+u"\n").encode('utf-8'))

    print 'parsing :', sentence
    # we will use lexicon knowledge to replace the 
    # unknown word in order to do a parsing with large corpus
    # of unknown words
    unknowns = check_unknown_words(tokens, grammar)
    if len(unknowns) > 0:
        grammar = update_grammar(productions, unknowns)
        parser = ViterbiParser(grammar) 
        parser.trace(2)

    parses = parser.parse_all(tokens)
    if len(parses) > 0:
        parse = parses[0]
    else:
        parse = ""
    test_output_file.write(" ".join(parse.__str__().replace("\n", '').split()) )
    test_output_file.write('\n')

    parses_bank.append(parse)

test_file.close()
test_output_file.close()

if args.run_in_shell != "True":
    sys.exit(0)

# run in shell
reload(sys)  
sys.setdefaultencoding('utf8')
#import codecs
#sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
sys.stdout.write(">>")
sentence = sys.stdin.readline().strip()
while sentence != "exit":
    tokens = word_tokenize(sentence)
    unknowns = check_unknown_words(tokens, grammar)
    if len(unknowns) > 0:
        grammar = update_grammar(productions, unknowns)
        parser = ViterbiParser(grammar) 
        parser.trace(2)

    parses = parser.parse_all(tokens)
    if len(parses) > 0:
        parse = parses[0]
        parse.un_chomsky_normal_form()  # show original form tree
    else:
        parse = ""
    sys.stdout.write("output parsing result: \n" + parse.__str__() + "\n")
    sys.stdout.write(">>")
    sentence = sys.stdin.readline().strip()
    
