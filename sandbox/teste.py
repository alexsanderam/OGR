import numpy as np
from scipy import ndimage
from skimage.morphology import medial_axis, skeletonize


def binary_skeletonization(A):
	return skeletonize(A).astype(np.uint8)


def binary_medial_axis(A):
	skel, dist = medial_axis(A, return_distance=True)
	return skel.astype(np.uint8)


def binary_hit_or_miss_transform(A, B=None, D=None):
	if B is None:
		B = np.ones((3,3))

	output = ndimage.binary_hit_or_miss(A, structure1=B).astype(np.uint8)
	return output


def binary_erosion(A, B=None):
	output = ndimage.binary_erosion(A, B)
	return output.astype(np.uint8)


def binary_dilation(A, B=None):
	output = ndimage.binary_dilation(A, B)
	return output.astype(np.uint8)