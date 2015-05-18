import numpy as np

def power_law(pixels, c, gamma):
	height = pixels.shape[0]
	width = pixels.shape[1]

	output = np.zeros((height, width))

	'''
	for x in range(0, height):
		for y in range(0, width):
				output[x, y] = np.power(np.dot(c, pixels[x, y]), gamma)
	'''
	output = np.power(np.dot(c, pixels), gamma)

	return output.astype(np.uint8)


def avg(pixels, order):
	mask = np.ones((order, order))
	return generic_filter(pixels, mask)


def median(pixels, order):
	height = pixels.shape[0]
	width = pixels.shape[1]

	offset = order / 2
		
	#output = np.zeros((height, width))
	output = np.copy(pixels)

	pos = (order*order) / 2

	for x in range(offset, height - offset):
		for y in range(offset, width - offset):
			
			neighbors = pixels[x-offset:x+offset+1, y-offset:y+offset+1]
			output[x, y] = np.sort(neighbors.reshape(neighbors.shape[0]*neighbors.shape[1]))[pos]

	return output.astype(np.uint8)



def generic_filter(pixels, mask):
	height = pixels.shape[0]
	width = pixels.shape[1]

	order = mask.shape[0]
	offset = order / 2

	#output = np.zeros((height, width))
	output = np.copy(pixels).astype(float)

	for x in range(offset, height - offset):
		for y in range(offset, width - offset):
					
			output[x, y] = 0
			divisor = 0

			for i in range(0, order):
				for j in range(0, order):
					output[x, y] = np.add(output[x, y], np.dot(mask[i, j], pixels[x+i-offset, y+j-offset]))
					divisor = np.add(divisor, mask[i, j])

			output[x, y] = np.divide(output[x, y], divisor)

	for x in range(0, offset - 1):
		for y in range(0, offset - 1):
			output[x, y] = pixels[x, y]

	return output.astype(np.uint8)


def gaussian_1D(pixels, order, sigma):
	height = pixels.shape[0]
	width = pixels.shape[1]

	offset = order / 2

	mask_x = np.zeros((order))
	mask_y = np.zeros((order))

	#output = np.zeros((height, width))
	output = np.copy(pixels).astype(float)
	
	for x in range(0, order):
		mask_x[x] = np.exp(-(np.power(x - offset, 2)) / (2*np.power(sigma, 2)).astype(float)) / (2 * np.pi * np.power(sigma, 2))

	for y in range(0, order):
		mask_y[y] = np.exp(-(np.power(y - offset, 2)) / (2*np.power(sigma, 2)).astype(float)) / (2 * np.pi * np.power(sigma, 2))	

	for x in range(offset, height - offset):
		for y in range(offset, width - offset):
			output[x, y] = 0
			for i in range(0, order):
				output[x, y] = np.add(output[x, y], np.dot(mask_y[i], pixels[x+i-offset, y]))
		
	for x in range(offset, height - offset):
		for y in range(offset, width - offset):
			output[x, y] = 0
			for j in range(0, order):
				output[x, y] = np.add(output[x, y], np.dot(mask_x[j], pixels[x, y+j-offset]))

	'''
	for x in range(0, offset - 1):
		for y in range(0, offset - 1):
			output[x, y] = pixels[x, y]
	'''

	return output.astype(np.uint8)


def gaussian_2D(pixels, order, sigma):
	height = pixels.shape[0]
	width = pixels.shape[1]

	offset = order / 2

	mask = np.zeros((order, order))

	#output = np.zeros((height, width))
	output = np.copy(pixels).astype(float)
	
	
	for x in range(0, order):
		for y in range(0, order):
			mask[x, y] = np.exp(-(np.power(x - offset, 2) + np.power(y - offset, 2)) / (2*np.power(sigma, 2)).astype(float)) / (2 * np.pi * np.power(sigma, 2))

	
	for x in range(offset, height - offset):
		for y in range(offset, width - offset):
			
			output[x, y] = 0

			for i in range(0, order):
				for j in range(0, order):
					#output[x, y] += mask[i, j] * pixels[x+i-offset, y+j-offset]
					output[x, y] = np.add(output[x, y], np.dot(mask[i, j], pixels[x+i-offset, y+j-offset]))

	return output.astype(np.uint8)
	

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
					#output[x, y] += mask[i, j] * pixels[x+i-offset, y+j-offset]
					output[x, y] = np.add(output[x, y], np.dot(mask[i, j], pixels[x+i-offset, y+j-offset]))

			output[x, y] = np.divide(output[x, y], 351)

	return output.astype(np.uint8)


def sobel(pixels):
	height = pixels.shape[0]
	width = pixels.shape[1]

	order = 3
	offset = order / 2

	mask_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
	mask_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

	grad_x = np.zeros((height, width))
	grad_y = np.zeros((height, width))

	#output = np.zeros((height, width))
	output = np.copy(pixels).astype(float)
	
	for x in range(offset, height - offset):
		for y in range(offset, width - offset):
			'''
			grad_x[x, y] += mask_x[0, 0] * pixels[x+0-offset, y+0-offset]
			grad_y[x, y] += mask_y[0, 1] * pixels[x+0-offset, y+0-offset]
			grad_x[x, y] += mask_x[0, 2] * pixels[x+0-offset, y+1-offset]
			grad_y[x, y] += mask_y[0, 0] * pixels[x+0-offset, y+1-offset]	
			grad_x[x, y] += mask_x[0, 1] * pixels[x+0-offset, y+2-offset]
			grad_y[x, y] += mask_y[0, 2] * pixels[x+0-offset, y+2-offset]
			grad_x[x, y] += mask_x[1, 0] * pixels[x+1-offset, y+0-offset]
			grad_y[x, y] += mask_y[1, 1] * pixels[x+1-offset, y+0-offset]
			grad_x[x, y] += mask_x[1, 2] * pixels[x+1-offset, y+1-offset]
			grad_y[x, y] += mask_y[1, 0] * pixels[x+1-offset, y+1-offset]
			grad_x[x, y] += mask_x[1, 1] * pixels[x+1-offset, y+2-offset]
			grad_y[x, y] += mask_y[1, 2] * pixels[x+1-offset, y+2-offset]
			grad_x[x, y] += mask_x[2, 0] * pixels[x+2-offset, y+0-offset]
			grad_y[x, y] += mask_y[2, 1] * pixels[x+2-offset, y+0-offset]
			grad_x[x, y] += mask_x[2, 2] * pixels[x+2-offset, y+1-offset]
			grad_y[x, y] += mask_y[2, 0] * pixels[x+2-offset, y+1-offset]
			grad_x[x, y] += mask_x[2, 1] * pixels[x+2-offset, y+2-offset]
			grad_y[x, y] += mask_y[2, 2] * pixels[x+2-offset, y+2-offset]
			'''
			output[x, y] = 0

			for i in range(0, order):
				for j in range(0, order):
					grad_x[x, y] = np.add(grad_x[x, y], np.dot(mask_x[i, j], pixels[x+i-offset, y+j-offset]))
					grad_y[x, y] = np.add(grad_y[x, y], np.dot(mask_y[i, j], pixels[x+i-offset, y+j-offset]))
			
			D = np.sqrt(np.power(grad_x[x,y], 2) + np.power(grad_y[x,y], 2))
			output[x, y] = D
	
	return output.astype(np.uint8)
