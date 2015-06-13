
class Circular_list(list):

	def __init__(self, M=25):
		self._index = 0
		self._l = []
		self._M = M


	def insert(self, x):
		if len(self._l) < self._M:
			self._l.append(x)
		else:
			self._l[self._index] = x

		self._index = (self._index + 1) % self._M


	def extend(self, l):
		for x in l:
			self.insert(x)
			

	def get_list(self):
		return self._l
	
	
	def clear(self):
		del self._l[:]
		self._index = 0


	def __str__(self):
		return str(self._l)


	def __eq__(self, y):
		return self._l == y._l


	def __len__(self):
		return len(self._l)

