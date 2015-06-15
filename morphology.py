import numpy as np


"""
Input: Set of binary pixels A, and structuring element B (optional).
Output: Result of erosion of A using the structuring element B (if given, otherwise using a default structuring element).
Obs: This implementation consider correspondence with the background, and don't care positions (expressed by negative values). Borders are ignored.
"""
def binary_erosion(A, B=None):
	height = A.shape[0]
	width = A.shape[1]
	output = np.zeros((height, width))
	
	if B is None:
		B = np.ones((3,3))

	order = B.shape[0]
	offset = order / 2

	for x in range(offset, height - offset):
		for y in range(offset, width - offset):

			i = 0
			match = 1
			while match and i < order:
				j = 0
				while match and j < order:
					if B[i, j] >= 0 and B[i, j] != A[x+i-offset, y+j-offset]:
						match = 0
					j += 1
				i += 1

			output[x, y] = match

	return output


"""
Input: Set of binary pixels A, and structuring element B (optional).
Output: Result of dilation of A using the structuring element B (if given, otherwise using a default structuring element).
Obs: This implementation consider correspondence with the background and don't care positions. Borders are ignored.
"""
def binary_dilation(A, B=None):
	height, width = A.shape
	output = np.zeros((height, width))

	if B is None:
		B = np.ones((3,3))

	order = B.shape[0]
	offset = order / 2

	for x in range(offset, height - offset):
		for y in range(offset, width - offset):

			A_and_B = np.logical_and(A[x-offset:x+offset+1, y-offset:y+offset+1], B)
			belonging = np.count_nonzero(A_and_B) > 0
			output[x, y] = belonging

	return output


"""
Input: Set of binary pixels A, structuring element B (optional) and number of iterations k (if not specified, k = 1 by default).
Output: Result of k erosions of A using the structuring element B and k dilations of the result of previous step using B.
		Opening of A using using the structuring element B, with k iterations.
"""
def binary_opening(A, B=None, k=1):
	output = np.copy(A)

	if B is None:
		B = np.ones((3,3))

	for i in range(0, k):
		output = binary_erosion(output, B)
	
	for i in range(0, k):
		output = binary_dilation(output, B)

	return output



"""
Input: Set of binary pixels A, structuring element B (optional) and number of iterations k (if not specified, k = 1 by default).
Output: Result of k dilations of A using the structuring element B and k erosions of the result of previous step using B.
		Closing of A using using the structuring element B, with k iterations.
"""
def binary_closing(A, B=None, k=1):
	output = np.copy(A)

	if B is None:
		B = np.ones((3,3))

	for i in range(0, k):
		output = binary_dilation(output, B)
	
	for i in range(0, k):
		output = binary_erosion(output, B)

	return output
	

"""
Input:  Set of binary pixels A.
Output: Result of skeletonization of A by Zhang and Suen Algorithm, and applying an m-connectivity algorithm.
"""
def zhang_and_suen_binary_thinning(A):
	height = A.shape[0]
	width = A.shape[1]

	_A = np.copy(A)
	
	removed_points = []
	flag_removed_point = True

	while flag_removed_point:

		flag_removed_point = False

		for x in range(1, height - 1):
			for y in range(1, width - 1):

				if _A[x,y] == 1:
					#get 8-neighbors
					neighborhood = [_A[x-1, y], _A[x-1, y+1], _A[x, y+1], _A[x+1, y+1], _A[x+1, y], _A[x+1, y-1], _A[x, y-1], _A[x-1, y-1]]
					P2, P3, P4, P5, P6, P7, P8, P9 = neighborhood

					#B_P1 is the number of nonzero neighbors of P1=(x, y)
					B_P1 = np.sum(neighborhood)
					condition_1 = 2 <= B_P1 <= 6
				
					#A_P1 is the number of 01 patterns in the ordered set of neighbors
					n = neighborhood + neighborhood[0:1]
					A_P1 = sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))

					condition_2 = A_P1 == 1
					condition_3 = P2 * P4 * P6 == 0
					condition_4 = P4 * P6 * P8 == 0

					if(condition_1 and condition_2 and condition_3 and condition_4):
						removed_points.append((x, y))
						flag_removed_point = True

		for x, y in removed_points:
			_A[x, y] = 0
		del removed_points[:]


		for x in range(1, height - 1):
			for y in range(1, width - 1):

				if _A[x,y] == 1:
					#get 8-neighbors
					neighborhood =  [_A[x-1, y], _A[x-1, y+1], _A[x, y+1], _A[x+1, y+1], _A[x+1, y], _A[x+1, y-1], _A[x, y-1], _A[x-1, y-1]]
					P2, P3, P4, P5, P6, P7, P8, P9 = neighborhood

					#B_P1 is the number of nonzero neighbors of P1=(x, y)
					B_P1 = np.sum(neighborhood)
					condition_1 = 2 <= B_P1 <= 6
				
					#A_P1 is the number of 01 patterns in the ordered set of neighbors
					n = neighborhood + neighborhood[0:1]
					A_P1 = sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))

					condition_2 = A_P1 == 1			
					condition_3 = P2 * P4 * P8 == 0
					condition_4 = P2 * P6 * P8 == 0
				
					if(condition_1 and condition_2 and condition_3 and condition_4):
						removed_points.append((x, y))
						flag_removed_point = True

		for x, y in removed_points:
			_A[x, y] = 0
		del removed_points[:]

	output = m_connectivity(_A)

	return output


"""
Input:  Set of binary pixels A.
Output: Result of m-connectivity of A.
"""
def m_connectivity(A):
	height, width = A.shape

	for x in range(1, height - 1):
		for y in range(1, width - 1):

			if A[x, y] == 1:

				d_1 = (x > 2) and (A[x-2, y-1] == 0 or A[x-2, y] == 1 or A[x-1, y-1] == 1)
				d_2 = (y > 2) and (A[x+1, y-2] == 0 or A[x, y-2] == 1 or A[x+1, y-1] == 1)
				d_3 = (y < width - 2) and (A[x-1, y+2] == 0 or A[x, y+2] == 1 or A[x-1, y+1] == 1)
				d_4 = (y < width - 2) and (A[x+1, y+2] == 0 or A[x, y+2] == 1 or A[x+1, y+1] == 1)

				if A[x-1, y+1] == 1 and (A[x-1, y] == 1 and d_1):
					A[x-1, y] = 0
				if A[x-1, y-1] == 1 and (A[x, y-1] == 1 and d_2):
					A[x, y-1] = 0
				if A[x+1, y+1] == 1 and (A[x, y+1] == 1 and d_3):
					A[x+1, y] = A[x, y+1] = 0
				if A[x-1, y+1] == 1 and (A[x, y+1] == 1 and d_4):
					A[x, y+1] = 0


	return A
