__author__ = 'aleks'
#from sentence import *

#!/usr/bin/env python

"""
A feature extractor for OTE
"""

# Separator of field values.
separator = ' '

# Field names of the input data.ou
# CONLL_STRUCTURE = ['id', 'orth', 'lemma', 'pos', 'pos2', 'morph', 'parent', 'relation', '_1', '_2', 'as' ]
#CONLL_STRUCTURE = ['id', 'orth', 'lemma', 'pos', 'pos2', 'morph', 'parent', 'relation', '_1', '_2', 'y' ]
CONLL_STRUCTURE = ['id', 'orth', 'lemma', 'pos', 'pos2', 'morph', 'parent', 'relation', 'POSLemma' ,  'isTopLemma',  'topLemma',  'sent', 'isPointedToByRule', 'ruleId',   'y' ]

fields = " ".join( CONLL_STRUCTURE )


# Attribute templates.
templates = (

    (('ruleId', 0), ),

    (('isPointedToByRule', 0), ),

    (('relation', -2), ),
    (('relation', -1), ),
    (('relation',  0), ),
    (('relation',  1), ),
    (('relation',  2), ),
    (('relation', -2), ('relation', -1)),
    (('relation', -1), ('relation',  0)),
    (('relation',  0), ('relation',  1)),
    (('relation',  1), ('relation',  2)),

    (('pos', -2), ),
    (('pos', -1), ),
    (('pos',  0), ),
    (('pos',  1), ),
    (('pos',  2), ),
    (('pos', -2), ('pos', -1)),
    (('pos', -1), ('pos',  0)),
    (('pos',  0), ('pos',  1)),
    (('pos',  1), ('pos',  2)),
    (('pos', -2), ('pos', -1), ('pos',  0)),
    (('pos', -1), ('pos',  0), ('pos',  1)),
    (('pos',  0), ('pos',  1), ('pos',  2)),
    )


import crfutils

def feature_extractor(X):
    # Apply attribute templates to obtain features (in fact, attributes)
    crfutils.apply_templates(X, templates)
    if X:
        # Append BOS and EOS features manually
        X[0]['F'].append('__BOS__')     # BOS feature
        X[-1]['F'].append('__EOS__')    # EOS feature

if __name__ == '__main__':
    crfutils.main(feature_extractor, fields=fields, sep=separator)
