from textgenrnn import textgenrnn
from datetime import datetime
import os

textgen = textgenrnn('mufon_longdescription_weights.hdf5')
    # this temperature schedule cycles between 1 very unexpected token, 1 unexpected token, 2 expected tokens, repeat.
# changing the temperature schedule can result in wildly different output!
temperature = [1.0, 0.5, 0.2]   
# temperature = [1.0]   

prefix = 'UFO REPORT: '   # if you want each generated text to start with a given seed text

# if train_cfg['line_delimited']:
#   n = 5
#   max_gen_length = 60 if model_cfg['word_level'] else 300
# else:
#   n = 1
#   max_gen_length = 2000 if model_cfg['word_level'] else 10000
  
timestring = datetime.now().strftime('%Y%m%d_%H%M%S')
gen_file = 'mufon_longdescription_weights.hdf5'

textgen.generate_to_file('mufon_longdescription_'+timestring+'gen.txt',
                         temperature=temperature,
                         prefix=prefix,
                         n=5,
                         max_gen_length=600)
