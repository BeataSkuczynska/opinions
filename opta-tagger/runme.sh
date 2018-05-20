# put your input file in input_data/conll-format, name it input.conll
# the file must be CONLL-formatted, contain morphological, POS and dependency tags. Order and number of columns must match
# the provided example, see input.conll file
# 
# The processing steps are as follows:
# (1) prepare data by identifying sentiment words; output CONLL format, append column with an "S" where sentiment found
#python2.7 add_sentiment.py input_data/conll-format/skladnica_test.conll input_data/conll-format/S_skladnica_test.conll
# (2) prepare data by seeking dependency patterns for opinion target extraction, append column to CONLL with rule_id (before sentiment column):
python2.7 opta_patterns.py input_data/conll-format/Sskladnica_test.conll input_data/conll-format/full_Sskladnica_test.conll
# (3) convert data to internal CRF Suite format:
cat input_data/conll-format/full_Sskladnica_test.conll | python2.7 crffeaturebuilder.py > input_data/crf-format/Sskladnica_test.crfsuite.txt
# (4) tag the data:
crfsuite-0.12/bin/crfsuite tag -m models/opta.model input_data/crf-format/Sskladnica_test.crfsuite.txt > input_data/crf-format/tagged_output_S_skladnica.conll
