import numpy as np
import util



def reflection(A):
	output = np.fliplr(A)
	output = output[::-1]

	return output	


def complement(A):
	return  1 - A


"""
def binary_erosion(A, B=None):
	height, width = A.shape

	output = np.zeros((height, width))
	
	order = 3#B.shape[0]
	offset = order / 2

	for x in range(offset, height - offset):
		for y in range(offset, width - offset):

			neighborhood = neighborhood = [A[x-1, y], A[x-1, y+1], A[x, y+1], A[x+1, y+1], A[x+1, y], A[x+1, y-1], A[x, y-1], A[x-1, y-1]]

			output[x, y] = np.min(neighborhood)

	return output
"""


def binary_erosion(A, B=None):
	height = A.shape[0]
	width = A.shape[1]
	output = np.zeros((height, width))
	
	if B is None:
		B = np.ones((3,3))

	order = B.shape[0]
	offset = order / 2
	contained = 0

	for x in range(offset, height - offset):
		for y in range(offset, width - offset):
	
			contained = 1
			i = 0

			while (contained and i < order):
				j = 0
				while (contained and j < order):

					if B[i, j] == 1 and A[x+i-offset, y+j-offset] == 0:
						contained = 0
					j = j + 1
				i = i + 1

			output[x, y] = contained
	return output


#dilation -> max
#border -> espelhar


def binary_dilation(A, B=None):
	height = A.shape[0]
	width = A.shape[1]
	output = np.zeros((height, width))

	if B is None:
		B = np.ones((3,3))

	order = B.shape[0]
	offset = order / 2
	belonging = 0

	for x in range(0, height):
		for y in range(0, width):
	
			belonging = 0

			if x < offset:
				i = offset - x
				u_i = order
			elif x >= height - offset:
				i = 0
				u_i = (x - (height - offset)) + 1
			else:
				i = 0
				u_i = order

			while (belonging == 0 and i < u_i):

				if y < offset:
					j = offset - y
					u_j = order
				elif y >= width - offset:
					j = 0
					u_j = (y - (width - offset)) + 1
				else:
					j = 0
					u_j = order

				while (belonging == 0 and j < u_j):

					if B[i, j] == 1 and A[x+i-offset, y+j-offset] == 1:
						belonging = 1

					j = j + 1

				i = i + 1

			output[x, y] = belonging
	return output


def binary_opening(A, B=None, k=1):
	output = np.copy(A)

	if B is None:
		B = np.ones((3,3))

	for i in range(0, k):
		output = binary_erosion(output, B)
	
	for i in range(0, k):
		output = binary_dilation(output, B)

	return output


def binary_closing(A, B=None, k=1):
	output = np.copy(A)

	if B is None:
		B = np.ones((3,3))

	for i in range(0, k):
		output = binary_dilation(output, B)
	
	for i in range(0, k):
		output = binary_erosion(output, B)

	return output


def binary_bondary_extraction(A, B=None):
	return A - binary_erosion(A, B)


def binary_hit_or_miss_transform(A, B=None, D=None):
	if B is None:
		B = np.ones((3,3))

	if D is None:
		#orderB = B.shape[0]
		#D = np.ones((orderB + 2, orderB + 2))
		#for i in range(1, D.shape[0] - 1):
		#	for j in range(1, D.shape[1] - 1):
		#		D[i, j] = 1 - B[i-1, j-1]
		#
		D = complement(B)

	temp1 = binary_erosion(A, B)
	temp2 = binary_erosion(complement(A), D)

	output = np.logical_and(temp1, temp2)

	return output.astype(np.uint8)


"""
def binary_hit_or_miss_transform(A, B=None, D=None):
	if B is None:
		B = np.ones((3,3))

	if D is None:
		#orderB = B.shape[0]
		#D = np.ones((orderB + 2, orderB + 2))
		#for i in range(1, D.shape[0] - 1):
		#	for j in range(1, D.shape[1] - 1):
		#		D[i, j] = 1 - B[i-1, j-1]
		
		D = complement(B)
		D_reflection = D
	else:
		D_reflection = reflection(D)
	

	temp1 = binary_erosion(A, B)
	temp2 = binary_dilation(A, D_reflection)

	output = np.subtract(temp1, temp2) #temp1 - temp2
	output = output.clip(0) #output[output < 0] = 0

	return output.astype(np.uint8)
"""
	

def binary_skeletonization(A, B=None):
	height, width = A.shape

	skel = np.zeros((height, width), dtype=np.uint8)
	S = []

	if B is None:
		#B = np.ones((3,3))
		B = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

	S.append(np.copy(A))

	k = 0
	while np.any(S[k]):
		k += 1
		temp = binary_erosion(S[k-1], B)
		S.append(temp)

	K = k - 1

	for k in range(0, K):
		S[k] = np.subtract(S[k], binary_opening(S[k], B))
		#S[k] = S[k].clip(0) #temp[temp < 0] = 0
		skel = np.logical_or(skel, S[k])

	output = (skel).astype(np.uint8)#m_connectivity(skel).astype(np.uint8)

	del S[:]

	return output

"""
def binary_skeletonization(A, B=None):
	height, width = A.shape
	
	skel = np.zeros((height, width), dtype=np.uint8)
	aux = A.copy()

	if B is None:
		#B = np.ones((3,3))
		B = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

	while np.any(aux):
		#temp = complement(binary_opening(aux))
		eroded = binary_erosion(aux)
		temp = binary_dilation(eroded)
		skel = np.logical_or(skel, np.logical_and(aux, temp))
		aux = eroded#binary_erosion(aux)

	output = (skel).astype(np.uint8)#m_connectivity(skel).astype(np.uint8)

	return output
"""

"""
def binary_thining(A, B=None,  D=None):
	temp = binary_hit_or_miss_transform(A, B, D)
	output = np.subtract(A, temp)
	output = output.clip(0) #output[output < 0] = 0	

	del temp
	return output
"""

#background operations are not required
def binary_thining(A, B=None):
	temp = binary_erosion(A, B)
	output = np.subtract(A, temp)
	output = output.clip(0) #output[output < 0] = 0	

	del temp
	return output


def binary_thining_by_sequence(A, S=None):
	height = A.shape[0]
	width = A.shape[0]

	output = np.zeros((height, width))
	
	if S is None:
		S = []
		B1 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
		S.append(B1)
		#B1 = np.array([[1, 1, 1], [0, 1, 0], [0, 0, 0]])
		#B1 = np.ones((3,3))
		#B2 = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]])#complement(B1)
		#S.append(B1)
		#S.append(B2)

		#for i in range(0, 3):
		#	_B1 = np.rot90(B1, 3 - i)
		#	_B2 = np.rot90(B2, 3 - i)
		#	S.append(_B1)
		#	S.append(_B2)

	result = A.copy()

	while not np.array_equal(output, result): #very slow
		output = result
		i = 0
		while i < len(S):
			result = binary_thining(result, S[i])
			i += 1
	output = m_connectivity(output)

	return output.astype(np.uint8)



def zang_and_suen_binary_thining(A):
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
					neighborhood = [_A[x-1, y], _A[x-1, y+1], _A[x, y+1], _A[x+1, y+1], _A[x+1, y], _A[x+1, y-1], _A[x, y-1], _A[x-1, y-1]]
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


def m_connectivity(A):
	height, width = A.shape
	#output = A#np.copy(A)

	for x in range(1, height - 1):
		for y in range(1, width - 1):

			#p_4_neighborhood = np.array([A[x+1, y], A[x-1, y], A[x, y+1], A[x, y-1]])
			#p_d_neighborhood = np.array([A[x+1, y+1], A[x+1, y-1], A[x-1, y+1], A[x-1, y-1]])
			#p_8_neighborhood = p_4_neighborhood + p_d_neighborhood

			#if A[x, y] == 1:
			#	if A[x+1, y+1] == 1 and (A[x+1, y] == 1 or A[x, y+1] == 1):
			#		A[x+1, y] = A[x, y+1] = 0
			#	if A[x+1, y-1] == 1 and (A[x+1, y] == 1 or A[x, y-1] == 1):
			#		A[x, y-1] = A[x+1, y] = 0
			#	if A[x-1, y+1] == 1 and (A[x-1, y] == 1 or A[x, y+1] == 1):
			#		A[x-1, y] = A[x, y+1] = 0
			#	if A[x-1, y-1] == 1 and (A[x-1, y] == 1 or A[x, y-1] == 1):
			#		A[x, y-1] = A[x-1, y] = 0

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
