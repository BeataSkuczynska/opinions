import pickle, sys, os
import codecs
from SentencePatternMatcher import *

paths_fname = 'resources/pathsByIdWithMetadata.pkl'

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print "usage: opta_patterns.py <input_filename> <output_filename>"

    input_fname = sys.argv[1]
    output_fname = sys.argv[2]

    if not os.path.isfile(input_fname):
        print "Input CONLL file is missing or is not readable"
        sys.exit(1)


    if not os.path.isfile(paths_fname):
        print "Dependency patterns file is missing or is not readable"
        sys.exit(1)

    spmatcher = SentencePatternMatcher()

    pathsByIdWithMetadata = pickle.load(open(paths_fname,'rb'))

    sentenceAsList = []
    with codecs.open(input_fname, 'r', 'utf8') as fr:
        with codecs.open(output_fname, 'w', 'utf8') as fw:

            for line in fr:
                if len(line) > 2:
                    sentenceAsList.append(line)
                else: # new sentence, parse what is loaded:

                    if len(sentenceAsList) > 0:
                        sentence = Sentence( sentenceAsList )
                        matchedTokensById = {}
                        startNodeIds = sentence.getTokenIDsWithAttribute('S')
                        for startNodeId in startNodeIds:
                            for id, p in pathsByIdWithMetadata.items():
                                tokenIDsFromRule = spmatcher.matchPathInSentence([startNodeId], p['path'] , sentence)
                                if tokenIDsFromRule is not None:
                                    for tokenId in tokenIDsFromRule:
                                        matchedTokensById[tokenId] = id

                        sentenceWithRuleId = []
                        for sline in sentenceAsList:
                            toks = sline.rstrip().split(' ')
                            tokenId = int(toks[0])
                            if tokenId in matchedTokensById:
                                toks.append( 'R' )
                                toks.append( str(matchedTokensById[tokenId]) )
                            else:
                                toks.append( '_' )
                                toks.append( '_' )
                            toks.append('\n')
                            sentenceWithRuleId.append( ' '.join(toks))

                        for out_line in sentenceWithRuleId:
                            fw.write(out_line)
                        fw.write('\n')

                    sentenceAsList = []




