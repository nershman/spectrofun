import sys
import argparse

from spectrofun import spectrofun as sp
#import numpy as np 

class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __eq__(self, other):
        return self.start <= other <= self.end

class OrderedNamespace(argparse.Namespace):
	#maintains order of all options in a list thing.
    def __init__(self, **kwargs):
        self.__dict__["_arg_order"] = []
        self.__dict__["_arg_order_first_time_through"] = True
        argparse.Namespace.__init__(self, **kwargs)

    def __setattr__(self, name, value):
        #print("Setting %s -> %s" % (name, value))
        self.__dict__[name] = value
        if name in self._arg_order and hasattr(self, "_arg_order_first_time_through"):
            self.__dict__["_arg_order"] = []
            delattr(self, "_arg_order_first_time_through")
        self.__dict__["_arg_order"].append(name)

    def _finalize(self):
        if hasattr(self, "_arg_order_first_time_through"):
            self.__dict__["_arg_order"] = []
            delattr(self, "_arg_order_first_time_through")

    def _latest_of(self, k1, k2):
        try:
            print(self._arg_order)
            if self._arg_order.index(k1) > self._arg_order.index(k2):
                return k1
        except ValueError:
            if k1 in self._arg_order:
                return k1
        return k2


parser = argparse.ArgumentParser(
				prog = 'FrequencyOffseter',
				description = 'offset frequency / spectral content, as well as invert',
				epilog = 'sneed')
parser.add_argument('infile')
parser.add_argument('outfile')
#the remaining arguments should be an "effects chain" but we can work on that later i guess
parser.add_argument('-f', '--offset', action='append', type=float, choices=[Range(0,100)], nargs=1) 
parser.add_argument('-i', '--invert', action='append_const', const=None)

if __name__ == '__main__':
#If I made this class based this stuff woudl all be in the setup part.
	args =  parser.parse_args(namespace=OrderedNamespace())
	print(args)
	inf = args.infile
	outf = args.outfile
	data = 1 #stand in for audio file
	arg_order = args._arg_order
	arg_order.remove('infile')
	arg_order.remove('outfile')
	offset_args = args.offset

	#note that if invert is chosen then args will have [None] since no arg needed
	#this just checks the appropraite numbe rog arg orders and arguments match
	if args.offset is not None and args.invert is not None:
		assert len(args._arg_order) == len(args.offset + args.invert)
	elif args.offset is not None:
		assert len(args._arg_order) == len(args.offset)
	elif args.invert is not None:
		assert len(args._arg_order) == len(args.invert)

	obj = sp.SpectroAudio(infile=inf, ofile=outf)

#Iterate throught the arguments to process the audio file.	
	for thing in arg_order:

		if thing is 'offset':
			obj.offset(offset_args[0][0])
			del offset_args[0]
		elif thing is 'invert':
			obj.invert()

	obj.export()



#what this should do: invert, then offset by 10 then offset by 0.4 then offset by 12
	#(base) sma-mac:spectrofun sma$ python test_temp_cli.py blah blah --invert -f 10 -f 0.4 -i -f 12
#Namespace(infile='blah', outfile='blah', offset=[[10.0], [0.4], [12.0]], invert=[None, None])

#DONE: its keeping the order of offsets and order odf inverts, but not sure if its keeping correct order of them togetyher.
#https://stackoverflow.com/questions/9027028/argparse-argument-order