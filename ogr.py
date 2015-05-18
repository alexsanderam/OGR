import filters
import util
import morphology

import numpy as np

import graph

from PIL import Image
from time import time




def preprocessing(pixels):
	print "-------------Preprocessing phase--------------"
	
	#Gaussian filter
	print "Passing gaussian filter."
	pixels = filters.gaussian(pixels)

	#define threshold to binarization of image
	k = util.otsu_thresholding(pixels)
	print "Threshold used in the binarization: %d" % (k)
	#binarize image using the threshold k
	print "Getting binarized image."
	pixels = util.binarization(pixels, k)

	print "Closing operator."
	pixels = morphology.binary_closing(pixels)
	print "Opening operator."
	pixels = morphology.binary_opening(pixels)

	return pixels


def segmentation(pixels, k=5):
	print "-------------Segmentation phase--------------"
	
	print "Applying %d erosions to get vertices pixels." % (k)
	vertices = np.copy(pixels)
	for i in range(0, k):
		vertices = morphology.binary_erosion(vertices)

	print "Applying %d dilations to get 'original' size of vertices." % (k)
	for i in range(0, k):
		vertices = morphology.binary_dilation(vertices)

	print "Getting edges from complement of vertices pixels"
	edges = np.subtract(pixels, vertices)
	edges = edges.clip(0)
	
	return vertices, edges


def topology_recognition(pixels, vertices):
	print "-------------Topology recognition phase--------------"

	print "Skelonization image."
	skel = morphology.zang_and_suen_binary_thining(pixels)
	print "Edge classification."
	pm, pe, pp, pc = edge_classification(skel, vertices)

	return skel, pm, pe, pp, pc


def edge_classification(skel, vertices):
	height, width = skel.shape

	miscellaneous_pixels = []
	edge_pixels = []
	crossing_pixels = []
	port_pixels = []

	for x in range(1, height-1):
		for y in range(1, width-1):

			if skel[x,y] == 1 and vertices[x,y] == 0:
				#four neighborhood of pixel (x, y)
				#neighborhood = [skel[x-1, y], skel[x+1, y], skel[x, y-1], skel[x, y+1]]
				#eight neighborhood of pixel (x, y)
				skel_neighborhood = [skel[x-1, y], skel[x+1, y], skel[x, y-1], skel[x, y+1], skel[x+1, y+1], skel[x+1, y-1], skel[x-1, y+1], skel[x-1, y-1]]
				vertex_neighborhood = [vertices[x-1, y], vertices[x+1, y], vertices[x, y-1], vertices[x, y+1], vertices[x+1, y+1], vertices[x+1, y-1], vertices[x-1, y+1], vertices[x-1, y-1]]
				#n0 is the number of object pixels in 4-neighborhood of (x,y)
				n0 = np.sum(skel_neighborhood)

				if n0 < 2:
					miscellaneous_pixels.append((x,y))
				elif n0 == 2 and np.any(vertex_neighborhood):
					port_pixels.append((x,y))
				elif n0 == 2:
					edge_pixels.append((x,y))
				else:
					crossing_pixels.append((x,y))

	return miscellaneous_pixels, edge_pixels, port_pixels, crossing_pixels


def postprocessing():
	print "-------------Posprocessing phase--------------"
	return None


def read_optical_graph(path):
	image = util.convert_to_gray_scale(util.read_image(path))
	return util.get_np_pixels(image)


def convert_to_topological_graph(pixels):
	start_wall_time = time()

	#====================================================================================
	#get start time of preprocessing phase
	start_time = time()

	#call preprocessing phase
	pixels = preprocessing(pixels)

	#get end time of preprocessing phase
	end_time = time()
	spent_time = (end_time - start_time)
	print "Time spent in the preprocessing phase: %.4f(s)\n" % (spent_time)

	#Visualization
	#util.show_image_from_binary_array(pixels, "Binarized image")
	#====================================================================================


	#====================================================================================
	#get start time of segmentation phase
	start_time = time()

	#call segmentation phase
	vertices, edges = segmentation(pixels)

	#get end time of preprocessing phase
	end_time = time()
	spent_time = (end_time - start_time)
	print "Time spent in the segmentation phase: %.4f(s)\n" % (spent_time)

	#Visualization
	#rgb = util.convert_binary_arrays_to_single_RGB_array(vertices, np.zeros((pixels.shape)), edges)
	#util.show_image_from_RGB_array(rgb, "Segmented")
	#util.show_image_from_binary_array(vertices, "Vertices")
	#====================================================================================


	#====================================================================================
	#get start time of topology recognition phase
	start_time = time()

	#call segmentation phase
	#pm: miscellaneous pixels, pe: edge pixels, pp: port pixels, pc: crossing pixels
	skel, pm, pe, pp, pc = topology_recognition(pixels, vertices)

	#get end time of preprocessing phase
	end_time = time()
	spent_time = (end_time - start_time)
	print "Time spent in the topology recognition phase: %.4f(s)\n" % (spent_time)

	#Visualization
	#edge_pixels = np.zeros((pixels.shape), dtype=np.uint8)
	#crossing_pixels = np.zeros((pixels.shape), dtype=np.uint8)
	#port_pixels = np.zeros((pixels.shape), dtype=np.uint8)
	#for x, y in pe:
	#	edge_pixels[x,y] = 1
	#for x, y in pc:
	#	crossing_pixels[x,y] = 1
	#for x, y in pp:
	#	port_pixels[x,y] = 1
	#rgb = util.convert_binary_arrays_to_single_RGB_array(port_pixels, crossing_pixels, edge_pixels)
	
	u#til.show_image_from_RGB_array(rgb, "Segmented")
	#util.show_image_from_binary_array(skel, "Skelonization")
	#====================================================================================



	end_time = time()
	spent_time = (end_time - start_wall_time)
	print "Total (wall) time spent: %.4f(s)\n" % (spent_time)

	return None


def save_topological_graph(topological_graph, path):
	#export_json(topological_graph, path)
	#export_tikz_pgf(topological_graph, path)
	return None


def export_tikz_pgf(graph, path):
	return None
