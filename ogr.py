import filters
import util
import morphology

import numpy as np

import graph

from PIL import Image
from time import time

from circular_list import *
from collections import Counter


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
	vertices_pixel = np.copy(pixels)
	for i in range(0, k):
		vertices_pixel = morphology.binary_erosion(vertices_pixel)

	print "Applying %d dilations to get 'original' size of vertices." % (k)
	for i in range(0, k):
		vertices_pixel = morphology.binary_dilation(vertices_pixel)
	
	return vertices_pixel



def get_connected_components_bfs_search(vertices_pixel):
	height, width = vertices_pixel.shape
	
	array_connected_components = np.zeros((height, width))
	connected_components = {}
	
	label = 1
	Q = []

	for x in range(1, height - 1):
		for y in range(1, width - 1):

			if vertices_pixel[x, y] == 0 or array_connected_components[x, y] != 0:
				continue

			array_connected_components[x, y] = label
			connected_components[label] = []
			connected_components[label].append((x, y))

			Q.append((x, y))

			while len(Q) > 0:
				p = Q[0]
				Q.remove(p)

				_x, _y = p[0], p[1]
				p_8_neighborhood = [[_x-1, _y-1], [_x-1, _y], [_x-1, _y+1], [_x, _y-1], [_x, _y+1], [_x+1, _y-1], [_x+1, _y], [_x+1, _y+1]]

				for q in p_8_neighborhood:

					i, j = q[0], q[1]

					if vertices_pixel[i, j] == 1 and array_connected_components[i, j] == 0:
						Q.append(q)
						array_connected_components[i, j] = label
						connected_components[label].append((i, j))

			label += 1

	return array_connected_components, connected_components



def topology_recognition(pixels, vertices_pixel):
	print "-------------Topology recognition phase--------------"

	print "Skelonization image."
	skel = morphology.zang_and_suen_binary_thinning(pixels)
	#skel = teste.binary_skeletonization(pixels)
	#skel = teste.binary_medial_axis(pixels)
	#skel = morphology.binary_skeletonization_by_thinning(pixels)

	print "Edge classification."
	classified_pixels, port_pixels, crossing_pixels = edge_classification(skel, vertices_pixel)

	#print "Edges section identify."
	trivial_sections, port_sections, crossing_sections = edge_sections_identify(classified_pixels, port_pixels, crossing_pixels)

	return skel, classified_pixels



def edge_classification(skel, vertices_pixel):
	height, width = skel.shape

	classified_pixels = np.zeros((skel.shape))
	port_pixels = []
	crossing_pixels = []

	for x in range(1, height-1):
		for y in range(1, width-1):

			if skel[x,y] == 1 and vertices_pixel[x,y] == 0:

				#eight neighborhood of pixel (x, y)
				skel_neighborhood = get_eight_neighborhood(skel, x, y)

				vertex_neighborhood = get_eight_neighborhood(vertices_pixel, x, y)
				vertex_neighborhood_in_skel = np.logical_and(skel_neighborhood, vertex_neighborhood)

				#n0 is the number of object pixels in 8-neighborhood of (x,y)
				n0 = np.sum(skel_neighborhood)

				if n0 < 2:
					classified_pixels[x,y] = 1 #miscellaneous pixels
				elif n0 == 2 and np.any(vertex_neighborhood_in_skel):
					classified_pixels[x,y] = 4 #port pixels
					port_pixels.append((x, y))
				elif n0 == 2:
					classified_pixels[x,y] = 2 #edge pixels
				elif n0 > 2:
					classified_pixels[x,y] = 3 #crossing pixels
					crossing_pixels.append((x, y))

	return classified_pixels, port_pixels, crossing_pixels


# def edge_classification(skel, vertices_pixel):
# 	height, width = skel.shape

# 	classified_pixels = np.zeros((skel.shape))
# 	port_pixels = []
# 	crossing_pixels = []

# 	for x in range(1, height-1):
# 		for y in range(1, width-1):

# 			if skel[x,y] == 1 and vertices_pixel[x,y] == 0:

# 				#eight neighborhood of pixel (x, y)
# 				skel_neighborhood = get_four_neighborhood(skel, x, y)

# 				#n0 is the number of object pixels in 8-neighborhood of (x,y)
# 				n0 = sum(skel_neighborhood.values())

# 				if n0 < 2:
# 					classified_pixels[x,y] = 1 #miscellaneous pixels
# 				elif n0 == 2 and n_vertex_pixel_in_neighborhood(skel, skel_neighborhood, vertices_pixel) > 0:
# 					classified_pixels[x,y] = 4 #port pixels
# 					port_pixels.append((x, y))
# 				elif n0 == 2:
# 					classified_pixels[x,y] = 2 #edge pixels
# 				elif n0 > 2:
# 					classified_pixels[x,y] = 3 #crossing pixels
# 					crossing_pixels.append((x, y))
			

# 	return classified_pixels, port_pixels, crossing_pixels



# def get_four_neighborhood(pixels, x, y):
# 	neighborhood = np.array([pixels[x-1, y], pixels[x+1, y], pixels[x, y-1], pixels[x, y+1]])
# 	return neighborhood



# def get_diagonal_neighborhood(pixels, x, y):
# 	neighborhood = np.array([pixels[x-1, y-1], pixels[x+1, y-1], pixels[x-1, y+1], pixels[x+1, y+1]])
# 	return neighborhood


def get_eight_neighborhood(pixels, x, y):
	neighborhood = np.array([pixels[x-1, y], pixels[x+1, y], pixels[x, y-1], pixels[x, y+1], pixels[x+1, y+1], pixels[x+1, y-1], pixels[x-1, y+1], pixels[x-1, y-1]])
	return neighborhood



# def get_four_neighborhood(pixels, x, y):
# 	neighborhood = {}
# 	neighborhood[x-1, y] = pixels[x-1, y]
# 	neighborhood[x+1, y] = pixels[x+1, y]
# 	neighborhood[x, y-1] = pixels[x, y-1]
# 	neighborhood[x, y+1] = pixels[x, y+1]

# 	return neighborhood


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



# def n_vertex_pixel_in_neighborhood(skel, neighborhood, vertices):
# 	n = 0
# 	for pos in neighborhood:
# 		if skel[pos] == 1 and vertices[pos] == 1:
# 			n += 1
# 	return n


def edge_sections_identify(classified_pixels, port_pixels, crossing_pixels):
	trivial_sections = []
	port_sections = []
	crossing_sections = []

	start_pixels = {}
	start_pixels = dict.fromkeys(port_pixels, 0)

	crossing_pixels_in_port_sectins = {}

	#dictionary of predecessors de crossing pixels
	back_port_sections = {}

	for start in start_pixels:

		#if port pixel is already visited, then continue
		if start_pixels[start] == 1:
			continue
		else:
			#marks start pixel as already visited
			start_pixels[start] = 1
			section, last_gradients, next = get_basic_section(start, classified_pixels)
		
		next_value = classified_pixels[next[0], next[1]]

		if next_value == 4: #port pixel
			section.append(next)

			#marks the next pixel as already visited
			start_pixels[(next[0], next[1])] = 1

			trivial_sections.append(section)

		elif next_value == 3: #crossing pixel
			#get last element in section before crossing pixel
			#back = section[-1][0], section[-1][1]
			
			section.append(next)
			port_sections.append(section)

			pos = (next[0], next[1])
			#if not pos in back_port_sections:
			#	back_port_sections[pos] = []

			#back_port_sections[pos].append(back)

			if not pos in crossing_pixels_in_port_sectins:
				crossing_pixels_in_port_sectins[pos] = []

			crossing_pixels_in_port_sectins[pos].append(section)

	#clear dictionary of start pixels
	start_pixels.clear()
	#start_pixels = dict.fromkeys(crossing_pixels_in_port_sectins, 0)

	#counter gradients frequency
	cnt_gradient = Counter(last_gradients.get_list())
	#count in list
	grads = cnt_gradient.most_common()

	for crossing_pixel in crossing_pixels_in_port_sectins:
		for section in crossing_pixels_in_port_sectins[crossing_pixel]:

			back = section[-1][0], section[-1][1]

			next_value = 0
			i = 0
			
			while next_value != 2: #edge pixel

				while next_value < 2 and i < len(grads):
					delta = grads[i][0]

					next = np.add(crossing_pixel, delta)

					if next[0] == back[0] and next[1] == back[1]:
						next_value = 0
					else:
						next_value = classified_pixels[next[0], next[1]]

					i += 1


				if next_value < 2 and i == len(grads):
					delta = get_gradient(classified_pixels, grads)
					next = np.add(crossing_pixel, delta)
					next_value = classified_pixels[next[0], next[1]]

				section.append(next)


			#back is an crossing pixel
			back = section[-1][0], section[-1][1]
			#pos = (back[0], back[1])

			if back in crossing_pixels_in_port_sectins:
				
				for _section in crossing_pixels_in_port_sectins[back]:
					if next[0] == _section[-2][0] and next[1] == _section[-2][1]:
							_section.extend(section[::-1][1:])
							#mark back as already visited
			else:
				_section, _last_gradients, next = get_basic_section(next, classified_pixels)

				#unnecessary
				_last_gradients.clear()
				del _last_gradients



	print len(trivial_sections), len(port_sections)
	return trivial_sections, port_sections, crossing_sections



def get_basic_section(start, classified_pixels):

	#'gradient' vector
	delta = np.array([0, 0])
	
	last_gradients = Circular_list()

	section = []
	section.append(start)

	x, y = start
	neighborhood_positions = np.array([[x-1, y-1], [x-1, y], [x-1,y+1], [x, y-1], [x, y+1], [x+1, y-1], [x+1, y], [x+1, y+1]])
	neighborhood = np.array([classified_pixels[x-1, y-1], classified_pixels[x-1, y], classified_pixels[x-1,y+1], classified_pixels[x, y-1], classified_pixels[x, y+1], classified_pixels[x+1, y-1], classified_pixels[x+1, y], classified_pixels[x+1, y+1]])

	next = neighborhood_positions[neighborhood.argmax()]
	next_value = classified_pixels[next[0], next[1]]
	delta = np.subtract(next, start)


	while next_value == 2: #edge pixel

		last_gradients.insert(delta)
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
						next = (x+i-1, y+j-1)
						next_value = classified_pixels[x+i-1, y+j-1]
			
			delta = np.subtract(next, last)

	last_gradients.insert(delta)
	last_element = next

	return section, last_gradients, last_element


def get_gradient(classified_pixels, grads):
	possible_grads = {[0, 1], [1, 0], [1, 1], [-1, 1], [-1, -1], [1, -1], [0, -1], [-1, 0]}
	most_common_grad = grads[0][0]
	possible_grads = possible_grads - set(grads)


	min_d = flot('inf')
	for grad in possible_grads:
		d = weighted_euclidean_distance(most_common_grad, grad)

		if d < min_d and classified_pixes[grad[0], grad[1]] > 1:
			min_grad = grad

	return min_grad


def weighted_euclidean_distance(grad1, grad2, alpha=2, betha=4):
	[x, y] = map(np.subtract, *zip(grad1, grad2))
	d = np.sqrt(alpha* (x**2) + betha * (y**2))
	return d


def traversal_subphase(classified_pixels, port_pixels, crossing_pixels, section):
	return None


def postprocessing():
	print "-------------Posprocessing phase--------------"
	return None


def get_vertices_coordinates(vertices_connected_components):
	vertices_coordinates = dict.fromkeys(vertices_connected_components, (0, 0))

	for label in vertices_connected_components:
		[x, y] =  map(sum, zip(*vertices_connected_components[label]))
		n = len(vertices_connected_components[label])
		vertices_coordinates[label]  = (x / n, y / n)

	return vertices_coordinates


def get_edges_coordinates(edges_section, k=10):
	n = np.count_nonzero(edges_section)
	step_size = n / k
	edges_sample = np.zeros((k, 2))

	for i in range(0, k):
		edges_sample[i] = edges_section[i * step_size]

	return edges_sample


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
	vertices_pixel = segmentation(pixels)

	#get end time of preprocessing phase
	end_time = time()
	spent_time = (end_time - start_time)
	print "Time spent in the segmentation phase: %.4f(s)\n" % (spent_time)

	#Visualization
	util.show_image_from_binary_array(vertices_pixel, "Vertices")

	#array_connected_components, connected_components = get_connected_components_bfs_search(vertices_pixel)
	#vertices_coordinates = get_vertices_coordinates(connected_components)
	#====================================================================================


	#====================================================================================
	#get start time of topology recognition phase
	start_time = time()

	#call segmentation phase
	skel, classified_pixels = topology_recognition(pixels, vertices_pixel)

	#get end time of preprocessing phase
	end_time = time()
	spent_time = (end_time - start_time)
	print "Time spent in the topology recognition phase: %.4f(s)\n" % (spent_time)

	# #Visualization
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