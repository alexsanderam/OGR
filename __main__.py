import os
import getopt
import sys

import numpy as np

import teste
import ogr

def __main__(argv):
	
	input_path = ''
	output_path = ''

	try:
		opts, args = getopt.getopt(argv, "i:o:", ["input=", "output="])
	except getopt.GetoptError:
		print('Illegal arguments')
		sys.exit(2)

	for opt, arg in opts:
		if opt in ('-h', '--help'):
			print('-i <input_image_path> -o <output_path>')
			sys.exit()
		elif opt in ("-i", "--input"):
			input_path = arg
		elif opt in ("-o", "--output"):
			output_path = arg

	print "\n"

	pixels = ogr.read_optical_graph(input_path)
	ogr.convert_to_topological_graph(pixels)

	#if output_path != '':
	#	util.save_image(output_image, output_path)
	#
	#output_image.show()		
	

if __name__ == "__main__":
	__main__(sys.argv[1:])
