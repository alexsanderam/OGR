import filters
import util
import morphology

import numpy as np

import graph

from PIL import Image
from time import time


import teste




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
	
	return vertices


def topology_recognition(pixels, vertices):
	print "-------------Topology recognition phase--------------"

	print "Skelonization image."
	#skel = morphology.zang_and_suen_binary_thining(pixels)
	#skel = teste.binary_skeletonization(pixels)
	#skel = teste.binary_medial_axis(pixels)
	skel = morphology.binary_skeletonization_by_thining(pixels)

	print "Edge classification."
	classified_pixels, port_pixels, crossing_pixels = edge_classification(skel, vertices)

	#print "Edges section identify."
	#trivial_sections, port_sections, crossing_sections = edge_sections_identify(classified_pixels, port_pixels, crossing_pixels)

	return skel, classified_pixels


# def edge_classification(skel, vertices):
# 	height, width = skel.shape

# 	classified_pixels = np.zeros((skel.shape))
# 	port_pixels = []
# 	crossing_pixels = []

# 	for x in range(1, height-1):
# 		for y in range(1, width-1):

# 			if skel[x,y] == 1 and vertices[x,y] == 0:

# 				#eight neighborhood of pixel (x, y)
# 				skel_neighborhood = morphology.get_eight_neighborhood(skel, x, y)

# 				vertex_neighborhood = morphology.get_eight_neighborhood(vertices, x, y)
# 				vertex_neighborhood_in_skel = np.multiply(skel_neighborhood, vertex_neighborhood)

# 				#n0 is the number of object pixels in 8-neighborhood of (x,y)
# 				n0 = np.sum(skel_neighborhood)

# 				if n0 < 2:
# 					classified_pixels[x,y] = 1 #miscellaneous pixels
# 				elif n0 == 2 and np.any(vertex_neighborhood_in_skel):
# 					classified_pixels[x,y] = 4 #port pixels
# 					port_pixels.append((x, y))
# 				elif n0 == 2:
# 					classified_pixels[x,y] = 2 #edge pixels
# 				elif n0 > 2:
# 					classified_pixels[x,y] = 3 #crossing pixels
# 					crossing_pixels.append((x, y))

# 	return classified_pixels, port_pixels, crossing_pixels


def edge_classification(skel, vertices):
	height, width = skel.shape

	classified_pixels = np.zeros((skel.shape))
	port_pixels = []
	crossing_pixels = []

	for x in range(1, height-1):
		for y in range(1, width-1):

			if skel[x,y] == 1 and vertices[x,y] == 0:

				#eight neighborhood of pixel (x, y)
				skel_neighborhood = get_four_neighborhood(skel, x, y)

				#n0 is the number of object pixels in 8-neighborhood of (x,y)
				n0 = sum(skel_neighborhood.values())

				if n0 < 2:
					classified_pixels[x,y] = 1 #miscellaneous pixels
				elif n0 == 2 and n_vertex_pixel_in_neighborhood(skel, skel_neighborhood, vertices) > 0:
					classified_pixels[x,y] = 4 #port pixels
					port_pixels.append((x, y))
				elif n0 == 2:
					classified_pixels[x,y] = 2 #edge pixels
				elif n0 > 2:
					classified_pixels[x,y] = 3 #crossing pixels
					crossing_pixels.append((x, y))
			

	return classified_pixels, port_pixels, crossing_pixels



# def get_four_neighborhood(pixels, x, y):
# 	neighborhood = np.array([pixels[x-1, y], pixels[x+1, y], pixels[x, y-1], pixels[x, y+1]])
# 	return neighborhood


# def get_diagonal_neighborhood(pixels, x, y):
# 	neighborhood = np.array([pixels[x-1, y-1], pixels[x+1, y-1], pixels[x-1, y+1], pixels[x+1, y+1]])
# 	return neighborhood


# def get_eight_neighborhood(pixels, x, y):
# 	neighborhood = np.array([pixels[x-1, y], pixels[x+1, y], pixels[x, y-1], pixels[x, y+1], pixels[x+1, y+1], pixels[x+1, y-1], pixels[x-1, y+1], pixels[x-1, y-1]])
# 	return neighborhood



def get_four_neighborhood(pixels, x, y):
	neighborhood = {}
	neighborhood[x-1, y] = pixels[x-1, y]
	neighborhood[x+1, y] = pixels[x+1, y]
	neighborhood[x, y-1] = pixels[x, y-1]
	neighborhood[x, y+1] = pixels[x, y+1]

	return neighborhood


# def get_diagonal_neighborhood(pixels, x, y):
# 	neighborhood = {}
# 	neighborhood[x-1, y-1] = pixels[x-1, y-1]
# 	neighborhood[x+1, y-1] = pixels[x+1, y-1]
# 	neighborhood[x-1, y+1] = pixels[x-1, y+1]
# 	neighborhood[x+1, y+1] = pixels[x+1, y+1]

# 	return neighborhood


# def get_eight_neighborhood(pixels, x, y):
# 	neighborhood = {}
# 	neighborhood[x-1, y] = pixels[x-1, y]
# 	neighborhood[x+1, y] = pixels[x+1, y]
# 	neighborhood[x, y-1] = pixels[x, y-1]
# 	neighborhood[x, y+1] = pixels[x, y+1]
# 	neighborhood[x+1, y+1] = pixels[x+1, y+1]
# 	neighborhood[x+1, y-1] = pixels[x+1, y-1]
# 	neighborhood[x-1, y+1] = pixels[x-1, y+1]
# 	neighborhood[x-1, y-1] = pixels[x-1, y-1]

# 	return neighborhood



def n_vertex_pixel_in_neighborhood(skel, neighborhood, vertices):
	n = 0
	for pos in neighborhood:
		if skel[pos] == 1 and vertices[pos] == 1:
			n += 1
	return n



def edge_sections_identify(classified_pixels, port_pixels, crossing_pixels):
	trivial_sections = []
	port_sections = []
	crossing_sections = []

	start_pixels = []
	start_pixels.extend(port_pixels)
	#start_pixels.extend(crossing_pixels)

	delta = np.array([0, 0])

	for start in start_pixels:

		section = []
		section.append(start)

		x, y = start
		positions = np.array([[x-1, y-1], [x-1, y], [x-1,y+1], [x, y-1], [x, y+1], [x+1, y-1], [x+1, y], [x+1, y+1]])
		neighborhood = np.array([classified_pixels[x-1, y-1], classified_pixels[x-1, y], classified_pixels[x-1,y+1], classified_pixels[x, y-1], classified_pixels[x, y+1], classified_pixels[x+1, y-1], classified_pixels[x+1, y], classified_pixels[x+1, y+1]])

		next = positions[neighborhood.argmax()]
		next_value = classified_pixels[next[0], next[1]]
		delta = np.subtract(next, start)


		while next_value == 2: #edge pixel
			section.append(next)

			next = np.add(next, delta)
			next_value = classified_pixels[next[0], next[1]]


			if next_value < 2: #blank pixel or miscellaneous pixel
				last = section[-1] #get last element added in section
				x, y =  last
				back = np.subtract(last, delta)

				#get max value in the neighborhood, unless the 'back'
				next_value  = -1
				for i in range(0, 3):
					for j in range(0, 3):

						if (x+i-1 != back[0] or y+j-1 != back[1]) and (i != 1 or j != 1) and (classified_pixels[x+i-1, y+j-1] > next_value):
							next = [x+i-1, y+j-1]
							next_value = classified_pixels[x+i-1, y+j-1]
				
				delta = np.subtract(next, last)

		
		if next_value == 4: #port pixel
			section.append(next)
			trivial_sections.append(section)
			#start_pixels.remove(next)
		elif next_value == 3: #crossing pixel
			section.append(next)
			port_sections.append(section)


	print len(trivial_sections), len(port_sections)


	return trivial_sections, port_sections, crossing_sections



def traversal_subphase(classified_pixels, port_pixels, crossing_pixels, section):
	return None


def postprocessing():
	print "-------------Posprocessing phase--------------"
	return None


def read_optical_graph(path):
	image = util.convert_to_gray_scale(util.read_image(path))
	return util.get_np_pixels(image)



def save_topological_graph(topological_graph, path):
	#export_json(topological_graph, path)
	#export_tikz_pgf(topological_graph, path)
	return None



def convert_to_topological_graph(pixels, name=None):
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
	#vertices, edges = segmentation(pixels)
	vertices = segmentation(pixels)

	#get end time of preprocessing phase
	end_time = time()
	spent_time = (end_time - start_time)
	print "Time spent in the segmentation phase: %.4f(s)\n" % (spent_time)

	#Visualization
	util.show_image_from_binary_array(vertices, "Vertices")
	#====================================================================================


	#====================================================================================
	#get start time of topology recognition phase
	start_time = time()

	#call segmentation phase
	skel, classified_pixels = topology_recognition(pixels, vertices)

	#get end time of preprocessing phase
	end_time = time()
	spent_time = (end_time - start_time)
	print "Time spent in the topology recognition phase: %.4f(s)\n" % (spent_time)

	#Visualization
	edge_pixels = np.zeros((pixels.shape), dtype=np.uint8)
	crossing_pixels = np.zeros((pixels.shape), dtype=np.uint8)
	port_pixels = np.zeros((pixels.shape), dtype=np.uint8)
	for x in range(classified_pixels.shape[0]):
		for y in range(classified_pixels.shape[1]):
		 	if classified_pixels[x, y] == 2:
		 		edge_pixels[x, y] = 1
		 	elif classified_pixels[x, y] == 3:
		 		crossing_pixels[x,y] = 1
		 	elif classified_pixels[x, y] == 4:
		 		port_pixels[x,y] = 1

	rgb = util.convert_binary_arrays_to_single_RGB_array(port_pixels, crossing_pixels, edge_pixels)
	
	#util.show_image_from_RGB_array(rgb, "Segmented")
	#util.show_image_from_binary_array(skel, "Skelonization")

	if not name is None:
		util.save_image_from_RGB_array(rgb, "../Resultados/Parcial/" + name + "_segmented.png")
		util.save_image_from_binary_array(skel, "../Resultados/Parcial/" + name + "_skel.png")
	#====================================================================================



	end_time = time()
	spent_time = (end_time - start_wall_time)
	print "Total (wall) time spent: %.4f(s)\n" % (spent_time)

	return None