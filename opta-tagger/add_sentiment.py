# (1) prepare data by identifying sentiment words; output CONLL format, append column with an "S" where sentiment found
# python add_sentiment.py input_data/conll-format/input.conll input_data/conll-format/S_input.conll
import sys, os
import codecs

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print "usage: add_sentiment.py <input_filename> <output_filename>"

    input_fname = sys.argv[1]
    output_fname = sys.argv[2]

    default_sent_fname = 'resources/slownikWydzwieku01.csv'

    if not os.path.isfile(input_fname):
        print "Input CONLL file is missing or is not readable"
        sys.exit(1)

    if not os.path.isfile(default_sent_fname):
        print "Default sentiment dictionary file is missing or is not readable"
        sys.exit(1)


    auto_senti = []
    with codecs.open(default_sent_fname,'r','utf8') as fr:
        for line in fr:
            toks = line.split('\t')
            if toks[2]=='1':
                auto_senti.append(toks[0])

    print "loaded",len(auto_senti),"words with sentiment"

    with codecs.open(input_fname,'r','utf8') as fr:
        with codecs.open(output_fname,'w','utf8') as fw:
            for line in fr:
                if len(line)>3:
                    toks = line.rstrip().split(' ')
                    if toks[2] in auto_senti:  # toks[2] = lemma
                        toks.append('S')
                    else:
                        toks.append('_')
                    fw.write(' '.join(toks))
                fw.write('\n')



