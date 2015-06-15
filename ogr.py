import filters
import util
import morphology

import numpy as np

from PIL import Image
from time import time

from circular_list import *
from collections import Counter


import networkx as nx
import matplotlib.pyplot as plt


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
	
	array_connected_components = np.zeros((height, width), dtype=np.uint8)
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
	skel = morphology.zhang_and_suen_binary_thinning(pixels)

	print "Edge classification."
	classified_pixels, port_pixels = edge_classification(skel, vertices_pixel)

	print "Edges section identify."
	trivial_sections, port_sections, crossing_pixels_in_port_sections, last_gradients = edge_sections_identify(classified_pixels, port_pixels)
	merged_sections = traversal_subphase(classified_pixels, crossing_pixels_in_port_sections, last_gradients)
	edge_sections = trivial_sections + merged_sections
	dict_edge_sections = get_dict_edge_sections(edge_sections, vertices_pixel)

	return dict_edge_sections, skel, classified_pixels



def edge_classification(skel, vertices_pixel):
	height, width = skel.shape

	classified_pixels = np.zeros((skel.shape))
	port_pixels = []

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

	return classified_pixels, port_pixels#, crossing_pixels


def get_eight_neighborhood(pixels, x, y):
	neighborhood = np.array([pixels[x-1, y], pixels[x+1, y], pixels[x, y-1], pixels[x, y+1], pixels[x+1, y+1], pixels[x+1, y-1], pixels[x-1, y+1], pixels[x-1, y-1]])
	return neighborhood



def edge_sections_identify(classified_pixels, port_pixels):
	trivial_sections = []
	port_sections = []
	crossing_sections = []

	start_pixels = {}
	start_pixels = dict.fromkeys(port_pixels, 0)

	last_gradients = {}
	crossing_pixels_in_port_sections = {}

	#dictionary of predecessors de crossing pixels
	back_port_sections = {}

	for start in start_pixels:

		#if port pixel is already visited, then continue
		if start_pixels[start] == 1:
			continue
		else:
			#marks start pixel as already visited
			start_pixels[start] = 1
			section, last_gradients[start], next = get_basic_section(start, classified_pixels)
		
		next_value = classified_pixels[next[0], next[1]]

		if next_value == 4: #port pixel

			#marks the next pixel as already visited
			start_pixels[(next[0], next[1])] = 1

			trivial_sections.append(section)

		elif next_value == 3: #crossing pixel
			
			port_sections.append(section)

			pos = (next[0], next[1])

			if not pos in crossing_pixels_in_port_sections:
				crossing_pixels_in_port_sections[pos] = []

			crossing_pixels_in_port_sections[pos].append([section, 0])

	#clear dictionary of start pixels
	start_pixels.clear()

	return trivial_sections, port_sections, crossing_pixels_in_port_sections, last_gradients


def traversal_subphase(classified_pixels, crossing_pixels_in_port_sections, last_gradients):

	merged_sections = []

	for crossing_pixel in crossing_pixels_in_port_sections:
		for info_section in crossing_pixels_in_port_sections[crossing_pixel]:

			#if crossing pixel is already visited, then continue
			if info_section[1] == 1:
				continue

			section = info_section[0]
			start_pixel = crossing_pixel
			
			flag_found_section = False
			iteration = 0

			while not flag_found_section:

				crossing_section_direction = get_crossing_section_direction(classified_pixels, start_pixel, last_gradients[section[0]], section)

				flag_found_section = merge_sections(crossing_pixels_in_port_sections, section, crossing_section_direction, merged_sections)


				if not flag_found_section:

					if len(crossing_section_direction) > 1:
						#start_back is a crossing pixel
						start_back = crossing_section_direction[-2]
					else:
						start_back = start_pixel


					#next is an edge pixel
					next = crossing_section_direction[-1]

					_section, _last_gradients, next = get_basic_section(next, classified_pixels, start_back)
					crossing_section_direction.extend(_section[1:])

					_crossing_section_direction = get_crossing_section_direction(classified_pixels, _section[-1], last_gradients[section[0]], _section)
					crossing_section = crossing_section_direction + _crossing_section_direction

					flag_found_section = merge_sections(crossing_pixels_in_port_sections, section, crossing_section, merged_sections)

					if not flag_found_section:
						#start pixel is a crossing pixel
						start_pixel = crossing_section[-2]
						section = section + crossing_section

					#if iteration == 1:
					#	_last_gradients.clear()
					#	del _last_gradients
					#else:
					last_gradients[section[0]].extend(_last_gradients)

				iteration += 1

	return merged_sections


def merge_sections(crossing_pixels_in_port_sections, section, crossing_section, merged_sections):


	if len(crossing_section) > 1:
		#back is a crossing pixel
		back = crossing_section[-2]
	else:
		back = section[-1]


	key_back = (back[0], back[1])


	#next is an edge pixel
	next = crossing_section[-1]

	if not crossing_pixels_in_port_sections.has_key(key_back):
		return False

	for info_section in crossing_pixels_in_port_sections[key_back]:
		_section = info_section[0]

		if next[0] == _section[-2][0] and next[1] == _section[-2][1]:
				
				merged_sections.append(section + crossing_section + _section[::-1][1:])

				#mark back (crossing pixel) as already visited
				info_section[1] = 1

				#print ((section[0][0], section[0][1]), (crossing_section[0][0], crossing_section[0][1]), (crossing_section[-1][0], crossing_section[-1][1]), (_section[::-1][-1][0], _section[::-1][-1][1])), next

				return True

	return False


def get_crossing_section_direction(classified_pixels, crossing_pixel, last_gradients, section):
	#counter gradients frequency
	cnt_gradient = Counter(last_gradients.get_list())
	#count in list
	grads = cnt_gradient.most_common()

	crossing_section_direction = []

	next = crossing_pixel
	next_value = classified_pixels[next[0], next[1]]

	#back is a edge pixel
	back = section[-2][0], section[-2][1]

	#avoid local minima
	iterations = 0
	loop_grads = Circular_list(3)
	excluded_grad = None
	
	while next_value != 2: #edge pixel

		aux_value = 0
		i = 0

		if iterations == 3:
			list_loop_grads = loop_grads.get_list()
			excluded_grad = list_loop_grads[1]
			crossing_section_direction[:] = []
			iterations = 0


		while aux_value < 2 and i < len(grads): #blank pixel or miscellaneous and i < len
			
			if grads[i][0] == excluded_grad:
				continue

			delta = grads[i][0]
			aux = np.add(next, delta)

			if aux[0] == back[0] and aux[1] == back[1]: #back[0] >= 0 and back[1] >= 0 and 
				aux_value = 0
			else:
				aux_value = classified_pixels[aux[0], aux[1]]

			i += 1

		if aux_value < 2 and i == len(grads):
			delta = get_gradient(classified_pixels, back, next, grads, excluded_grad)
			loop_grads.insert(delta)
			back = next[0], next[1]
			next = np.add(next, delta)
			next_value = classified_pixels[next[0], next[1]]
		else:
			loop_grads.insert(delta)
			back = next[0], next[1]
			next = aux
			next_value = aux_value

		crossing_section_direction.append(next)
		iterations += 1

	return crossing_section_direction


def get_basic_section(start, classified_pixels, start_back=None):

	#'gradient' vector
	delta = np.array([0, 0])
	
	last_gradients = Circular_list()

	section = []
	section.append(start)

	x, y = start
	next, next_value = get_max_neighbor(classified_pixels, x, y, start_back)
	delta = np.subtract(next, start)


	while next_value == 2: #edge pixel

		last_gradients.insert((delta[0], delta[1]))
		section.append(next)

		next = np.add(next, delta)
		next_value = classified_pixels[next[0], next[1]]


		if next_value < 2: #blank pixel or miscellaneous pixel
			last = section[-1] #get last element added in section
			x, y =  last
			back = np.subtract(last, delta)

			#get max value in the neighborhood, unless the 'back'
			next, next_value = get_max_neighbor(classified_pixels, x, y, back)

			delta = np.subtract(next, last)

	last_gradients.insert((delta[0], delta[1]))
	section.append(next)
	last_element = next

	return section, last_gradients, last_element


#get max value in the neighborhood, unless the 'back'
def get_max_neighbor(classified_pixels, x, y, back=None):
	
	neighbor = None
	neighbor_value = -float('inf')

	for i in range(0, 3):
		for j in range(0, 3):
			if (back is None or (x+i-1 != back[0] or y+j-1 != back[1])) and (i != 1 or j != 1) and (classified_pixels[x+i-1, y+j-1] > neighbor_value):
				neighbor = np.array([x+i-1, y+j-1])
				neighbor_value = classified_pixels[x+i-1, y+j-1]

	return neighbor, neighbor_value



#change name of this function
def get_gradient(classified_pixels, back, current, common_grads, excluded_grad=None):

	possible_grads = {(0, 1), (1, 0), (1, 1), (-1, 1), (-1, -1), (1, -1), (0, -1), (-1, 0)}
	s_grads = [x[0] for x in common_grads]
	possible_grads = possible_grads - set(s_grads)

	if not excluded_grad is None:
		possible_grads = possible_grads - {excluded_grad}

	min_d = float('inf')
	min_grad = None

	for grad in possible_grads:
		aux = np.add(current, grad)

		sat_condition = (aux[0] != back[0] or aux[1] != back[1]) and (classified_pixels[aux[0], aux[1]] > 1)
	
		d = distance_heuristic_grads(common_grads, possible_grads, grad)

		if sat_condition and (d < min_d):
			min_d = d
			min_grad = grad

	return min_grad



def distance_heuristic_grads(common_grads, possible_grads, grad):
	
	n = 0.0
	average_common_grad = [0.0, 0.0]
	#most_common_grad = grads[0][0]

	for _grad in common_grads:
		aux =  [_grad[1] * z for z in _grad[0]]
		average_common_grad = map(sum, zip(average_common_grad, aux))
		n += _grad[1]

	average_common_grad = [z / n for z in average_common_grad]

	#determine weights to calculate distances
	amount_non_zero_x = 0
	amount_non_zero_y = 0 

	for _grad in common_grads:
		if _grad[0][0] != 0:
			amount_non_zero_x += _grad[1]
		if _grad[0][1] != 0:
			amount_non_zero_y += _grad[1]

	total_non_zeros = amount_non_zero_x + amount_non_zero_y

	alpha = (total_non_zeros - amount_non_zero_y)
	betha = (total_non_zeros - amount_non_zero_x)

	#print alpha, betha

	d = weighted_euclidean_distance(average_common_grad, grad, alpha, betha)
	#d = weighted_euclidean_distance(most_common_grad, grad, alpha, betha)
	#print amount_non_zero_x, amount_non_zero_y, "alpha: ", alpha, "betha: ", betha, "distance: ", d

	return d



def weighted_euclidean_distance(grad1, grad2, alpha=1, betha=1):
	[x, y] = [grad1[0] - grad2[0], grad1[1] - grad2[1]]
	d = np.sqrt((alpha* (x**2)) + (betha * (y**2)))
	return d


def get_dict_edge_sections(edge_sections, vertices_pixel):
	dict_edge_sections = {}

	for section in edge_sections:
		
		u_x, u_y = section[0]
		v_x, v_y = section[-1]

		u_vertex_neighbor_positions = [[u_x-1, u_y-1], [u_x-1, u_y], [u_x-1, u_y+1], [u_x, u_y-1], [u_x, u_y+1], [u_x+1, u_y-1], [u_x+1, u_y], [u_x+1, u_y+1]]
		v_vertex_neighbor_positions = [[v_x-1, v_y-1], [v_x-1, v_y], [v_x-1, v_y+1], [v_x, v_y-1], [v_x, v_y+1], [v_x+1, v_y-1], [v_x+1, v_y], [v_x+1, v_y+1]]

		pixel_u = (0, 0)
		pixel_v = (0, 0)

		for (i, j) in u_vertex_neighbor_positions:
			if vertices_pixel[i, j] > 0:
				pixel_u = (i, j)
				break

		for (i, j) in v_vertex_neighbor_positions:
			if vertices_pixel[i, j] > 0:
				pixel_v = (i, j)
				break

		dict_edge_sections[pixel_u, pixel_v] = section

	return dict_edge_sections



def postprocessing(vertices_pixel, dict_edge_sections):
	print "-------------Posprocessing phase--------------"
	
	G = nx.Graph()

	print "Getting vertices coordinates."
	array_connected_components, connected_components = get_connected_components_bfs_search(vertices_pixel)
	vertices_coordinates = get_vertices_coordinates(connected_components)


	print "Adding vertices in the graph."
	for u in connected_components:
		G.add_node(u, x=vertices_coordinates[u][0], y=vertices_coordinates[u][1])


	print "Getting edges extremes and adding edges in the graph."
	for (pos_u, pos_v) in dict_edge_sections:
		
		u = array_connected_components[pos_u[0], pos_u[1]]
		v = array_connected_components[pos_v[0], pos_v[1]]

		#edge_coordinates = get_edges_coordinates(dict_edge_sections[pos_u, pos_v])

		G.add_edge(u, v)#, Path=edge_coordinates)

	return G, vertices_coordinates


def get_vertices_coordinates(vertices_connected_components):
	vertices_coordinates = dict.fromkeys(vertices_connected_components, (0, 0))

	for label in vertices_connected_components:
		[x, y] =  map(sum, zip(*vertices_connected_components[label]))
		n = len(vertices_connected_components[label])
		vertices_coordinates[label]  = (x / n, y / n)

	return vertices_coordinates


def get_edges_coordinates(edge_section, k=10):
	n = len(edge_section)
	step_size = n / k
	edges_sample = []

	for i in range(0, k):
		edges_sample.append(edge_section[i * step_size])

	return edges_sample


def read_optical_graph(path):
	image = util.convert_to_gray_scale(util.read_image(path))
	return util.get_np_pixels(image)


def save_topological_graph(G, path):
	nx.write_graphml(G, path)


def convert_to_topological_graph(pixels, path = None, name=None):

	#pixels = np.rot90(pixels, 3)
	
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
	preprocessing_pixels = np.copy(pixels)
	#====================================================================================


	#====================================================================================
	#get start time of segmentation phase
	start_time = time()

	#call segmentation phase
	vertices_pixel = segmentation(pixels)

	#get end time of segmentation phase
	end_time = time()
	spent_time = (end_time - start_time)
	print "Time spent in the segmentation phase: %.4f(s)\n" % (spent_time)
	#====================================================================================


	#====================================================================================
	#get start time of topology recognition phase
	start_time = time()

	#call topology recognition phase
	dict_edge_sections, skel, classified_pixels = topology_recognition(pixels, vertices_pixel)

	#get end time of topology recognition phase
	end_time = time()
	spent_time = (end_time - start_time)
	print "Time spent in the topology recognition phase: %.4f(s)\n" % (spent_time)
	#====================================================================================


	#====================================================================================
	#get start time of posprocessing phase
	start_time = time()


	#call posprocessing phase
	G, vertices_coordinates = postprocessing(vertices_pixel, dict_edge_sections)

	#get end time of posprocessing phase
	end_time = time()
	spent_time = (end_time - start_time)
	print "Time spent in the topology recognition phase: %.4f(s)\n" % (spent_time)
	#====================================================================================



	end_time = time()
	spent_time = (end_time - start_wall_time)
	print "Total (wall) time spent: %.4f(s)\n" % (spent_time)



	#====================================================================================	
	#Resulting graph
	print "---------------------Resulting graph--------------------"
	print "Graph nodes: ", G.nodes()
	print "Graph edges: ", G.edges()

	if not (path is None or name is None):
		save_topological_graph(G, path + name + ".graphml")
		print "\nGraphml saved on path: " + path + name + ".graphml"
	#====================================================================================	


	#====================================================================================
	#Visualization

	if not (path is None or name is None):
		pos = {}#vertices_coordinates
		for x in vertices_coordinates:
			pos[x] = (vertices_coordinates[x][1], pixels.shape[0] - vertices_coordinates[x][0])

		colors=range(nx.number_of_edges(G))
		nx.draw(G, pos=pos, node_color='#A0CBE2',edge_color=colors, width=4, edge_cmap=plt.cm.winter,with_labels=False)
		plt.draw()
		plt.savefig(path + name + "_result.png")
		#plt.show()


		util.save_image_from_binary_array(preprocessing_pixels, path + name + "_preprocessing.png")

		util.save_image_from_binary_array(vertices_pixel, path + name + "_vertices_pixel.png")

		util.save_image_from_binary_array(skel, path + name + "_skel.png")

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
		rgb_edge_classification = util.convert_binary_arrays_to_single_RGB_array(port_pixels, crossing_pixels, edge_pixels)
		util.save_image_from_RGB_array(rgb_edge_classification, path + name + "_edge_classification.png")

	#====================================================================================

	return None
