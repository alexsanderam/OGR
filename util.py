from PIL import Image
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from pylab import *


def read_image(path):
	image = Image.open(path, 'r')
	return image


def convert_to_gray_scale(image):
	return image.convert('L')


def get_np_pixels(image):
	return np.asarray(image, dtype=np.uint8)


def get_pixels(image):
	return image.load()


def save_image(image, path):
	image.save(path)


def save_image_from_binary_array(pixels, path):
	image = convert_binary_array_to_image(pixels).convert('L')
	save_image(image, path)


def save_image_from_RGB_array(rgb, path):
	image = convert_array_to_image(rgb)
	save_image(image, path)


def show_image_from_binary_array(pixels, title=None):
	return convert_binary_array_to_image(pixels).show(title=title)


def show_image_from_RGB_array(rgb, title=None):
	convert_array_to_image(rgb).show(title=title)


def convert_binary_array_to_image(pixels):
	return convert_array_to_image((255*pixels))


def convert_array_to_image(pixels):
	return Image.fromarray(pixels)


def convert_binary_arrays_to_single_RGB_array(R, G, B):
	R = (255 * R)
	G = (255 * G)
	B = (255 * B)
	rgb = np.dstack((R, G, B))

	return rgb.astype(np.uint8)


"""
vertices, [255, 255, 0], edges, [255, 0, 0], port_pixels, [0, 0, 255], crossing_pixels, [0, 255, 0]
"""


def binarization(pixels, k):

	#height = pixels.shape[0]
	#width = pixels.shape[1]
	#output = np.zeros((height, width))
	output = (pixels <= k)

	return output.astype(np.uint8)


def negative(pixels):
		
	#height = pixels.shape[0]
	#width = pixels.shape[1]
	#output = np.zeros((height, width))

	output = 255 - pixels

	return output.astype(np.uint8)


def otsu_thresholding(pixels):
	normalized_histogram, bin_centers = get_positive_normalized_histogram_array(pixels)
	k, max_variance = histogram_based_global_thresholding(normalized_histogram, bin_centers)

	return k


def histogram_based_global_thresholding(normalized_histogram, bin_centers):
	P1, P2 = classes_probability(normalized_histogram)
	m1, m2 = medium_intensity(normalized_histogram, bin_centers, P1, P2)
	
	variance = variance_between_classes(P1, P2, m1, m2)
	idx = np.argmax(variance)
	max_k = bin_centers[idx]
	max_variance = variance[idx]
		
	return max_k, max_variance


def classes_probability(normalized_histogram):
	P1 = P2 = 0
	
	P1 = np.cumsum(normalized_histogram)
	P2 = np.cumsum(normalized_histogram[::-1])[::-1]

	return P1, P2


def medium_intensity(normalized_histogram, bin_centers, P1, P2):
	m1 = np.cumsum(bin_centers * normalized_histogram) / P1
	m2 = (np.cumsum((bin_centers * normalized_histogram)[::-1]) / P2[::-1])[::-1]

	return m1, m2


def variance_between_classes(P1, P2, m1, m2):
	variance = P1 * P2 * (m1 - m2)**2
	return variance	



#def histogram_based_global_thresholding(normalized_histogram, L):
#	max_variance = 0
#	max_k = 1
#
#	for k in range(1, L - 1):
#		P1, P2 = classes_probability(normalized_histogram, k)
#		m1, m2 = medium_intensity(normalized_histogram, L, P1, P2, k)
#	
#		variance = variance_between_classes(P1, P2, m1, m2)
#		
#		#mg = cumulative_medium_intensity(P1, P2, m1, m2)
#		#m = k_cumulative_medium_intensity(normalized_histogram, k)
#		#variance = variance_between_classes(P1, mg, m)
#
#		if variance > max_variance:
#			#print max_k, max_variance
#			max_variance = variance
#			max_k = k
#		#print k, variance, P1, mg, m
#
#	return max_k, max_variance


#def variance_between_classes(P1, P2, m1, m2):
#	variance = P1 * P2 * (m1 - m2)**2
#	return variance	


#def variance_between_classes(P1, mg, m):
#	variance = 0
#	if (P1 != 1) and (P1 != 0):
#		variance = np.power(mg*P1 - m, 2) / (P1*(1 - P1))
#	
#	return variance


#def classes_probability(normalized_histogram, k):
#	P1 = P2 = 0
#
#	#for i in range(0, k):
#	#	P1 += normalized_histogram[i]	
#	#P2 = 1 - P1
#	
#	P1 = np.cumsum(normalized_histogram[:k])[k-1]
#	P2 = 1 - P1
#
#	return P1, P2


#def medium_intensity(normalized_histogram, L, P1, P2, k):
#	m1 = m2 = 0
#		
#	if P1 > 0:
#		for i in range(0, k):
#			m1 += i * normalized_histogram[i]
#	
#		m1 /= P1
#
#	if P2 > 0:
#		for i in range(k + 1, L - 1):
#			m2 += i * normalized_histogram[i]
#		m2 /= P2
#
#	return m1, m2


#def k_cumulative_medium_intensity(normalized_histogram, k):
#	m = 0
#	for i in range(0, k):
#		m += i * normalized_histogram[i]	
#	return m


#def cumulative_medium_intensity(P1, P2, m1, m2):
#	mg = P1*m1 + P2*m2
#	return mg


def get_positive_normalized_histogram_array(pixels):
	output = get_normalized_histogram_array(pixels)
	bin_centers = np.nonzero(output)[0]
	output = output[bin_centers]

	return output, bin_centers


def get_histogram_array(pixels):
	height = pixels.shape[0]
	width = pixels.shape[1]

	output = np.zeros((256), dtype=np.uint64)

	for x in range(0, height):
		for y in range(0, width):
			output[pixels[x, y]] += 1

	return output


def get_normalized_histogram_array(pixels):
	height = pixels.shape[0]
	width = pixels.shape[1]

	return np.dot(1.0/(height * width), get_histogram_array(pixels))


def histogram_plot(histogram_array):
	plt.hist(range(0, 256), weights=histogram_array, bins=256)
	plt.xlabel('Intensity')
	plt.ylabel('Probability')
	plt.title(r'Histogram')
	plt.show()


def histogram_plot_from_pixels(pixels):
	figure()
	hist(pixels.flatten(), 256)
	plt.xlabel('Intensity')
	plt.ylabel('Probability')
	plt.title(r'Histogram')
	show()
