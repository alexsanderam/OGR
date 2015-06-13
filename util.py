from PIL import Image
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from pylab import *



"""
Input: image source path.
Output: image (PIL format).
"""
def read_image(path):
	image = Image.open(path, 'r')
	return image


"""
Input: RGB Image (PIL format).
Output: gray scale image (one channel).
"""
def convert_to_gray_scale(image):
	return image.convert('L')


"""
Input: image (PIL format).
Output: pixels of image in numpy array format.
"""
def get_np_pixels(image):
	return np.asarray(image, dtype=np.uint8)


"""
Input: image (PIL format).
Output: image pixels (PIL format).
"""
def get_pixels(image):
	return image.load()


"""
Input: image (PIL format).
Saves the image in the specified path.
"""
def save_image(image, path):
	image.save(path)


"""
Input: numpy binary array of pixels.
Saves the image related to the numpy array in the specified path.
"""
def save_image_from_binary_array(pixels, path):
	image = convert_binary_array_to_image(pixels).convert('L')
	save_image(image, path)


"""
Input: numpy rgb array of pixels.
Saves the image related to the numpy array in the specified path.
"""
def save_image_from_RGB_array(rgb, path):
	image = convert_array_to_image(rgb)
	save_image(image, path)


"""
Input: numpy binary array of pixels.
Displays the image related to the numpy array
"""
def show_image_from_binary_array(pixels, title=None):
	convert_binary_array_to_image(pixels).show(title=title)


"""
Input: numpy rgb array of pixels.
Displays the image related to the numpy array
"""
def show_image_from_RGB_array(rgb, title=None):
	convert_array_to_image(rgb).show(title=title)


"""
Input: numpy binary array of pixels.
Output: numpy gray scale array related to the input.
"""
def convert_binary_array_to_image(pixels):
	return convert_array_to_image((255*pixels))


"""
Input: numpy array of pixels.
Output: Image (PIL format) related to the numpy array.
"""
def convert_array_to_image(pixels):
	return Image.fromarray(pixels)


"""
Input: numpy binary arrays of red pixels, green pixels and blue pixels.
Output: single rgb numpy array.
"""
def convert_binary_arrays_to_single_RGB_array(R, G, B):
	R = (255 * R)
	G = (255 * G)
	B = (255 * B)
	rgb = np.dstack((R, G, B))

	return rgb.astype(np.uint8)


"""
Input: numpy gray scale array of pixels.
Output: binarized numpy array according to the threshold k (numpy array format).
"""
def binarization(pixels, k):

	#height = pixels.shape[0]
	#width = pixels.shape[1]
	#output = np.zeros((height, width))
	output = (pixels <= k)

	return output.astype(np.uint8)


"""
Input: numpy gray scale array of pixels.
Output: negative of pixels (numpy array format).
"""
def negative(pixels):
		
	#height = pixels.shape[0]
	#width = pixels.shape[1]
	#output = np.zeros((height, width))

	output = 255 - pixels

	return output.astype(np.uint8)


"""
Input: gray scale numpy array.
Output: best threshold k according to the otsu algorithm (based on the largest variance).
"""
def otsu_thresholding(pixels):
	normalized_histogram, bin_centers = get_positive_normalized_histogram_array(pixels)
	k, max_variance = histogram_based_global_thresholding(normalized_histogram, bin_centers)

	return k


"""
Input: normalized histogram, and bin centers of histogram.
Output: bin center that provides the largest variance in the histogram and their largest variance.
"""
def histogram_based_global_thresholding(normalized_histogram, bin_centers):
	P1, P2 = classes_probability(normalized_histogram)
	m1, m2 = medium_intensity(normalized_histogram, bin_centers, P1, P2)
	
	variance = variance_between_classes(P1, P2, m1, m2)
	idx = np.argmax(variance)
	max_k = bin_centers[idx]
	max_variance = variance[idx]
		
	return max_k, max_variance


"""
Input: normalized histogram.
Output: P1: probability of belonging to class referring to bin center (numpy array format)
		P2: probability of not belonging to the class regarding the bin center (numpy array format).
"""
def classes_probability(normalized_histogram):
	P1 = P2 = 0
	
	P1 = np.cumsum(normalized_histogram)
	P2 = np.cumsum(normalized_histogram[::-1])[::-1]

	return P1, P2


"""
Input: normalized histogram, bin centers and probability class (P1 and P2).
Output: m1: average intensity of class 1 (numpy array format)
		m2: average intensity of class 2 (numpy array format)
"""
def medium_intensity(normalized_histogram, bin_centers, P1, P2):
	m1 = np.cumsum(bin_centers * normalized_histogram) / P1
	m2 = (np.cumsum((bin_centers * normalized_histogram)[::-1]) / P2[::-1])[::-1]

	return m1, m2


"""
Input: probability class (P1 and P2) and average intensity of classes
Output: variance between classes (numpy array format)
"""
def variance_between_classes(P1, P2, m1, m2):
	variance = P1 * P2 * (m1 - m2)**2
	return variance	



"""
Input: gray scale numpy array.
Output: positive normalized histogram and their bin centers (numpy array format).
"""
def get_positive_normalized_histogram_array(pixels):
	output = get_normalized_histogram_array(pixels)
	bin_centers = np.nonzero(output)[0]
	output = output[bin_centers]

	return output, bin_centers


"""
Input: gray scale numpy array.
Output: histogram (numpy array format).
"""
def get_histogram_array(pixels):
	height = pixels.shape[0]
	width = pixels.shape[1]

	output = np.zeros((256), dtype=np.uint64)

	for x in range(0, height):
		for y in range(0, width):
			output[pixels[x, y]] += 1

	return output


"""
Input:gray scale numpy array.
Output: normalized histogram (numpy array format).
"""
def get_normalized_histogram_array(pixels):
	height = pixels.shape[0]
	width = pixels.shape[1]

	return np.dot(1.0/(height * width), get_histogram_array(pixels))


"""
Input: histogram numpy array.
	   Displays the histogram.
"""
def histogram_plot(histogram_array):
	plt.hist(range(0, 256), weights=histogram_array, bins=256)
	plt.xlabel('Intensity')
	plt.ylabel('Probability')
	plt.title(r'Histogram')
	plt.show()


"""
Input: gray scale numpy array.
	   Displays the histogram.
"""
def histogram_plot_from_pixels(pixels):
	figure()
	hist(pixels.flatten(), 256)
	plt.xlabel('Intensity')
	plt.ylabel('Probability')
	plt.title(r'Histogram')
	show()
