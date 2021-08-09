from textgenrnn import textgenrnn
from datetime import datetime
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--count', type=int, 
                      required=False,
                      default=10)
parser.add_argument('--length', type=int, 
                      required=False,
                      default=200)
parser.add_argument('--temperature', type=float, 
                      required=False,
                      default=1.0)
parser.add_argument('--prefix', type=str, 
                      required=False,
                      default='')
args = parser.parse_args()
textgen = textgenrnn('weights/mufon_longdescription_weights.hdf5')
    # this temperature schedule cycles between 1 very unexpected token, 1 unexpected token, 2 expected tokens, repeat.
# changing the temperature schedule can result in wildly different output!
# temperature = [1.0, 0.5, 0.2, 0.2]   
temperature = [1.0]   

prefix = "UFO AI REPORT: "   # if you want each generated text to start with a given seed text

# if train_cfg['line_delimited']:
#   n = 5
#   max_gen_length = 60 if model_cfg['word_level'] else 300
# else:
#   n = 1
#   max_gen_length = 2000 if model_cfg['word_level'] else 10000
  
timestring = datetime.now().strftime('%Y%m%d_%H%M%S')

textgen.generate_to_file('outputs/mufon_longdescription_gen_'+timestring+"_"+str(args.count)+"_"+str(args.length)+"_"+str(args.temperature).replace(".","_")+'txt',
                         temperature=args.temperature,
                         prefix=args.prefix,
                         n=args.count,
                         max_gen_length=args.length)
