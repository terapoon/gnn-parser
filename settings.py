UPOSTAG_IDX_DICT = {
    'ADJ': 1,       # adjective
    'ADP': 2,       # adposition
    'ADV': 3,       # adverb
    'AUX': 4,       # auxiliary verb
    'CCONJ': 5,      # coordinating conjunction
    'DET': 6,       # determiner
    'INTJ': 7,      # interjection
    'NOUN': 8,      # noun
    'NUM': 9,       # numeral
    'PART': 10,     # particle
    'PRON': 11,     # pronoun
    'PROPN': 12,    # proper noun
    'PUNCT': 13,    # punctuation
    'SCONJ': 14,    # subordinating conjunction
    'SYM': 15,      # symbol
    'VERB': 16,     # verb
    'X': 17,        # other
    'ROOT': 0
}


UPOSTAG_LABEL_DICT = {
    1: 'ADJ',       # adjective
    2: 'ADP',       # adposition
    3: 'ADV',       # adverb
    4: 'AUX',       # auxiliary verb
    5: 'CCONJ',     # coordinating conjunction
    6: 'DET',       # determiner
    7: 'INTJ',      # interjection
    8: 'NOUN',      # noun
    9: 'NUM',       # numeral
    10: 'PART',     # particle
    11: 'PRON',     # pronoun
    12: 'PROPN',    # proper noun
    13: 'PUNCT',    # punctuation
    14: 'SCONJ',    # subordinating conjunction
    15: 'SYM',      # symbol
    16: 'VERB',     # verb
    17: 'X',        # other
    0: 'ROOT'
}


DEPREL_IDX_DICT = {
    'acl': 1,           # clausal modifier of noun (adjectival clause)
    'advcl': 2,         # adverbial clause modifier
    'advmod': 3,        # adverbial modifier
    'amod': 4,          # adjectival modifier
    'appos': 5,         # appositional modifier
    'aux': 6,           # auxiliary
    'case': 7,          # case marking
    'cc': 8,            # coordinating conjunction
    'ccomp': 9,         # clausal complement
    'clf': 10,          # classifier
    'compound': 11,     # compound
    'conj': 12,         # conjunct
    'cop': 13,          # copula
    'csubj': 14,        # clausal subject
    'dep': 15,          # unspecified dependency
    'det': 16,          # determiner
    'discourse': 17,    # discourse element
    'dislocated': 18,   # dislocated elements
    'expl': 19,         # expletive
    'fixed': 20,        # fixed multiword expression
    'flat': 21,         # flat multiword expression
    'goeswith': 22,     # goes with
    'iobj': 23,         # indirect object
    'list': 24,         # list
    'mark': 25,         # marker
    'nmod': 26,         # nominal modifier
    'nsubj': 27,        # nominal subject
    'nummod': 28,       # numeric modifier
    'obj': 29,          # object
    'obl': 30,          # oblique nominal
    'orphan': 31,       # orphan
    'parataxis': 32,    # parataxis
    'punct': 33,        # punctuation
    'reparandum': 34,   # overridden disfluency
    'root': 35,         # root
    'vocative': 36,     # vocative
    'xcomp': 37,        # open clausal complement
    'nop': 0
}
