import numpy as np

import teste


def reflection(A):
	output = np.fliplr(A)
	output = output[::-1]

	return output	


def complement(A):
	return  1 - A



# def binary_erosion(A, B=None):
# 	return teste.binary_erosion(A, B)



# def binary_erosion(A, B=None):
# 	height = A.shape[0]
# 	width = A.shape[1]
# 	output = np.zeros((height, width))
	
# 	if B is None:
# 		B = np.ones((3,3))

# 	order = B.shape[0]
# 	offset = order / 2
# 	contained = 0

# 	for x in range(offset, height - offset):
# 		for y in range(offset, width - offset):
	
# 			contained = 1
# 			i = 0

# 			while (contained and i < order):
# 				j = 0
# 				while (contained and j < order):

# 					if B[i, j] == 1 and A[x+i-offset, y+j-offset] == 0:
# 						contained = 0
# 					j = j + 1
# 				i = i + 1

# 			output[x, y] = contained
# 	return output



# def binary_erosion(A, B=None):
# 	height = A.shape[0]
# 	width = A.shape[1]
# 	output = np.zeros((height, width))
	
# 	if B is None:
# 		B = np.ones((3,3))

# 	order = B.shape[0]
# 	offset = order / 2

# 	for x in range(offset, height - offset):
# 		for y in range(offset, width - offset):
# 			match = np.array_equal(A[x-offset:x+offset+1, y-offset:y+offset+1], B)
# 			output[x, y] = match

# 	return output


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




# def binary_dilation(A, B=None):
#  	return teste.binary_dilation(A, B)



# def binary_dilation(A, B=None):
# 	height = A.shape[0]
# 	width = A.shape[1]
# 	output = np.zeros((height, width))

# 	if B is None:
# 		B = np.ones((3,3))

# 	order = B.shape[0]
# 	offset = order / 2
# 	belonging = 0

# 	for x in range(0, height):
# 		for y in range(0, width):
	
# 			belonging = 0

# 			if x < offset:
# 				i = offset - x
# 				u_i = order
# 			elif x >= height - offset:
# 				i = 0
# 				u_i = (x - (height - offset)) + 1
# 			else:
# 				i = 0
# 				u_i = order

# 			while (belonging == 0 and i < u_i):

# 				if y < offset:
# 					j = offset - y
# 					u_j = order
# 				elif y >= width - offset:
# 					j = 0
# 					u_j = (y - (width - offset)) + 1
# 				else:
# 					j = 0
# 					u_j = order

# 				while (belonging == 0 and j < u_j):

# 					if B[i, j] == 1 and A[x+i-offset, y+j-offset] == 1:
# 						belonging = 1

# 					j = j + 1

# 				i = i + 1

# 			output[x, y] = belonging
# 	return output



# def binary_dilation(A, B=None):
# 	height, width = A.shape
# 	output = np.zeros((height, width))

# 	if B is None:
# 		B = np.ones((3,3))

# 	order = B.shape[0]
# 	offset = order / 2

# 	for x in range(offset, height - offset):
# 		for y in range(offset, width - offset):

# 			belonging = np.count_nonzero(A[x-offset:x+offset+1, y-offset:y+offset+1] == B) > 0
# 			output[x, y] = belonging

# 	return output



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



# def binary_hit_or_miss_transform(A, B=None, D=None):
# 	if B is None:
# 		B = np.ones((3,3))

# 	if D is None:
# 		#orderB = B.shape[0]
# 		#D = np.ones((orderB + 2, orderB + 2))
# 		#for i in range(1, D.shape[0] - 1):
# 		#	for j in range(1, D.shape[1] - 1):
# 		#		D[i, j] = 1 - B[i-1, j-1]
		
# 		D = complement(B)
# 		D_reflection = D
# 	else:
# 		D_reflection = reflection(D)
	

# 	temp1 = binary_erosion(A, B)
# 	temp2 = binary_dilation(A, D_reflection)

# 	output = np.subtract(temp1, temp2) #temp1 - temp2
# 	output = output.clip(0) #output[output < 0] = 0

# 	return output.astype(np.uint8)

	

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


# def binary_skeletonization(A, B=None):
# 	height, width = A.shape
	
# 	skel = np.zeros((height, width), dtype=np.uint8)
# 	aux = A.copy()

# 	if B is None:
# 		#B = np.ones((3,3))
# 		B = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

# 	while np.any(aux):
# 		#temp = complement(binary_opening(aux))
# 		eroded = binary_erosion(aux)
# 		temp = binary_dilation(eroded)
# 		skel = np.logical_or(skel, np.logical_and(aux, temp))
# 		aux = eroded#binary_erosion(aux)

# 	output = (skel).astype(np.uint8)#m_connectivity(skel).astype(np.uint8)

# 	return output



# def binary_thining(A, B=None,  D=None):
# 	temp = binary_hit_or_miss_transform(A, B, D)
# 	output = np.subtract(A, temp)
# 	output = output.clip(0) #output[output < 0] = 0	

# 	del temp
# 	return output


#background operations are not required
def binary_thining(A, B=None):
	temp = binary_erosion(A, B)
	output = np.subtract(A, temp)
	output = output.clip(0) #output[output < 0] = 0	

	del temp
	return output


# def binary_skeletonization_by_thining(A, S=None):
# 	height = A.shape[0]
# 	width = A.shape[0]

# 	output = np.zeros((height, width))
	
# 	if S is None:
# 		S = []

# 		# B1 = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 1]])
# 		# B2 = np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0]])
# 		# B3 = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 0]])
# 		# B4 = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
# 		# B5 = np.array([[1, 1, 1], [0, 1, 0], [0, 0, 0]])
# 		# B6 = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]])
# 		# B7 = np.array([[0, 0, 1], [0, 1, 1], [0, 0, 1]])
# 		# B8 = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]])

# 		# B1 = np.array([[0, 0, 0], [-1, 1, -1], [1, 1, 1]])
# 		# B2 = np.array([[-1, 0, 0], [1, 1, 0], [-1, 1, -1]])
# 		# B3 = np.array([[1, -1, 0], [1, 1, 0], [1, -1, 0]])
# 		# B4 = np.array([[-1, 1, -1], [1, 1, 0], [-1, 0, 0]])
# 		# B5 = np.array([[1, 1, 1], [-1, 1, -1], [0, 0, 0]])
# 		# B6 = np.array([[-1, 1, -1], [0, 1, 1], [0, 0, -1]])
# 		# B7 = np.array([[0, -1, 1], [0, 1, 1], [0, -1, 1]])
# 		# B8 = np.array([[0, 0, -1], [0, 1, 1], [-1, 1, -1]])

# 		B1 = np.array([[0, 0, 0], [-1, 1, -1], [1, 1, 1]])
# 		B2 = np.array([[-1, 0, 0], [1, 1, 0], [1, 1, -1]])
# 		B3 = np.array([[1, -1, 0], [1, 1, 0], [1, -1, 0]])
# 		B4 = np.array([[1, 1, -1], [1, 1, 0], [-1, 0, 0]])
# 		B5 = np.array([[1, 1, 1], [-1, 1, -1], [0, 0, 0]])
# 		B6 = np.array([[-1, 1, 1], [0, 1, 1], [0, 0, -1]])
# 		B7 = np.array([[0, -1, 1], [0, 1, 1], [0, -1, 1]])
# 		B8 = np.array([[0, 0, -1], [0, 1, 1], [-1, 1, 1]])



# 		S.append(B1)
# 		S.append(B2)
# 		S.append(B3)
# 		S.append(B4)
# 		S.append(B5)
# 		S.append(B6)
# 		S.append(B7)
# 		S.append(B8)

# 	result = A.copy()

# 	while not np.array_equal(output, result): #very slow
# 		output = result
		
# 		i = 0
# 		while i < len(S):
# 			result = binary_thining(result, S[i])
# 			i += 1

# 	return output.astype(np.uint8)


def binary_skeletonization_by_thining(A):
	height, width = A.shape

	output = np.zeros((height, width))
	
	B1 = np.array([[0, 0, 0], [-1, 1, -1], [1, 1, 1]])
	B2 = np.array([[-1, 0, 0], [1, 1, 0], [1, 1, -1]])
	B3 = np.array([[1, -1, 0], [1, 1, 0], [1, -1, 0]])
	B4 = np.array([[1, 1, -1], [1, 1, 0], [-1, 0, 0]])
	B5 = np.array([[1, 1, 1], [-1, 1, -1], [0, 0, 0]])
	B6 = np.array([[-1, 1, 1], [0, 1, 1], [0, 0, -1]])
	B7 = np.array([[0, -1, 1], [0, 1, 1], [0, -1, 1]])
	B8 = np.array([[0, 0, -1], [0, 1, 1], [-1, 1, 1]])

	result = A.copy()

	while not np.array_equal(output, result): #very slow
		output = result
		
		result = binary_thining(result, B1)
		result = binary_thining(result, B2)
		result = binary_thining(result, B3)
		result = binary_thining(result, B4)
		result = binary_thining(result, B5)
		result = binary_thining(result, B6)
		result = binary_thining(result, B7)
		result = binary_thining(result, B8)

	return result.astype(np.uint8)


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


	output = _A

	return output


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