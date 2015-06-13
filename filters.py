import numpy as np


def gaussian(pixels):
	height = pixels.shape[0]
	width = pixels.shape[1]

	order = 5
	offset = order / 2

	mask = np.array([[1, 5, 8, 5, 1], [5, 21, 34, 21, 5], [8, 34, 55, 34, 8], [5, 21, 34, 21, 5], [1, 5, 8, 5, 1]])

	#output = np.zeros((height, width))
	output = np.copy(pixels).astype(float)
	
	for x in range(offset, height - offset):
		for y in range(offset, width - offset):
			
			output[x, y] = 0

			for i in range(0, order):
				for j in range(0, order):
					output[x, y] += mask[i, j] * pixels[x+i-offset, y+j-offset]

			output[x, y] /=  351

	return output.astype(np.uint8)
