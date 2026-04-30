"""
Combination variant: all six optimisations applied together.
Changes applied:
  - fillFloatLumaFromBufferImage : vectorised np.asarray luma (v1)
  - boxFilter                    : integral image / summed-area table (v2)
  - dctOutput2hash               : np.median + vectorised threshold (v3 + v4)
  - decimateFloat                : np.arange + np.ix_ fancy indexing (v5)
  - dct64To16                    : NumPy @ matrix multiply (v6)
"""

import math

from PIL import Image

from matrix import MatrixUtil #Ensure that matrix.py is in the same directory as this file!
from imagehash import ImageHash
import numpy as np


class FINDHasher_all:

	#  From Wikipedia: standard RGB to luminance (the 'Y' in 'YUV').
	LUMA_FROM_R_COEFF = float(0.299)
	LUMA_FROM_G_COEFF = float(0.587)
	LUMA_FROM_B_COEFF = float(0.114)

	#  Since FINd uses 64x64 blocks, 1/64th of the image height/width
	#  respectively is a full block.
	FIND_WINDOW_SIZE_DIVISOR = 64

	def compute_dct_matrix(self):
		matrix_scale_factor = math.sqrt(2.0 / 64.0)
		d = [0] * 16
		for i in range(0, 16):
			di = [0] * 64
			for j in range(0, 64):
				di[j] = math.cos((math.pi / 2 / 64.0) * (i + 1) * (2 * j + 1))
			d[i] = di
		return d

	def __init__(self):
		"""See also comments on dct64To16. Input is (0..63)x(0..63); output is
		(1..16)x(1..16) with the latter indexed as (0..15)x(0..15).
		Returns 16x64 matrix."""
		self.DCT_matrix = self.compute_dct_matrix()

	def fromFile(self, filepath):
		img = None
		try:
			img = Image.open(filepath)
		except IOError as e:
			raise e
		return self.fromImage(img)

	def fromImage(self,img):
		try:
			# resizing the image proportionally to max 512px width and max 512px height
			img=img.copy()
			img.thumbnail((512, 512)) #https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail
		except IOError as e:
			raise e
		numCols, numRows = img.size
		buffer1 = MatrixUtil.allocateMatrixAsRowMajorArray(numRows, numCols)
		buffer2 = MatrixUtil.allocateMatrixAsRowMajorArray(numRows, numCols)
		buffer64x64 = MatrixUtil.allocateMatrix(64, 64)
		buffer16x64 = MatrixUtil.allocateMatrix(16, 64)
		buffer16x16 = MatrixUtil.allocateMatrix(16, 16)
		numCols, numRows = img.size
		self.fillFloatLumaFromBufferImage(img, buffer1)
		return self.findHash256FromFloatLuma(
			buffer1, buffer2, numRows, numCols, buffer64x64, buffer16x64, buffer16x16
		)

	# v1: vectorised luma
	def fillFloatLumaFromBufferImage(self, img, luma):
		numCols, numRows = img.size
		rgb_image = img.convert("RGB")
		numCols, numRows = img.size
		arr = np.asarray(rgb_image, dtype=np.float64)
		luma_2d = (
			self.LUMA_FROM_R_COEFF * arr[:, :, 0]
			+ self.LUMA_FROM_G_COEFF * arr[:, :, 1]
			+ self.LUMA_FROM_B_COEFF * arr[:, :, 2]
		)
		flat = luma_2d.ravel()
		for k in range(flat.size):
			luma[k] = float(flat[k])

	def findHash256FromFloatLuma(
		self,
		fullBuffer1,
		fullBuffer2,
		numRows,
		numCols,
		buffer64x64,
		buffer16x64,
		buffer16x16,
	):
		windowSizeAlongRows = self.computeBoxFilterWindowSize(numCols)
		windowSizeAlongCols = self.computeBoxFilterWindowSize(numRows)

		self.boxFilter(fullBuffer1,fullBuffer2,numRows,numCols,windowSizeAlongRows,windowSizeAlongCols)
		fullBuffer1=fullBuffer2

		self.decimateFloat(fullBuffer1, numRows, numCols, buffer64x64)
		self.dct64To16(buffer64x64, buffer16x64, buffer16x16)
		hash = self.dctOutput2hash(buffer16x16)
		return hash

	# v5: vectorised decimation
	@classmethod
	def decimateFloat(cls, in_, inNumRows, inNumCols, out):
		row_idx = (((np.arange(64) + 0.5) * inNumRows) / 64).astype(int)
		col_idx = (((np.arange(64) + 0.5) * inNumCols) / 64).astype(int)
		arr = np.array(in_, dtype=np.float64).reshape(-1, inNumCols)
		sampled = arr[np.ix_(row_idx, col_idx)]
		for i in range(64):
			for j in range(64):
				out[i][j] = float(sampled[i, j])

	# v6: matrix multiply DCT
	def dct64To16(self, A, T, B):
		""" Full 64x64 to 64x64 can be optimized e.g. the Lee algorithm.
		But here we only want slots (1-16)x(1-16) of the full 64x64 output.
		Careful experiments showed that using Lee along all 64 slots in one
		dimension, then Lee along 16 slots in the second, followed by
		extracting slots 1-16 of the output, was actually slower than the
		current implementation which is completely non-clever/non-Lee but
		computes only what is needed."""
		D = np.array(self.DCT_matrix, dtype=np.float64)   # 16x64
		A_arr = np.array(A, dtype=np.float64)             # 64x64
		T_arr = D @ A_arr                                  # 16x64
		B_arr = T_arr @ D.T                                # 16x16
		for i in range(16):
			for j in range(16):
				B[i][j] = float(B_arr[i, j])
		for i in range(16):
			for j in range(64):
				T[i][j] = float(T_arr[i, j])

	# v3+v4: np.median + vectorised threshold
	def dctOutput2hash(self, dctOutput16x16):
		"""
		Each bit of the 16x16 output hash is for whether the given frequency
		component is greater than the median frequency component or not.
		"""
		arr = np.array(dctOutput16x16, dtype=np.float64)
		dctMedian = float(np.median(arr))
		hash_bits = (arr > dctMedian).astype(int)
		hash_bits = hash_bits[::-1, ::-1]
		return ImageHash(hash_bits.reshape((256,)))

	@classmethod
	def computeBoxFilterWindowSize(cls, dimension):
		""" Round up."""
		return int(
			(dimension + cls.FIND_WINDOW_SIZE_DIVISOR - 1)
			/ cls.FIND_WINDOW_SIZE_DIVISOR
		)

	# v2: integral image box filter
	@classmethod
	def boxFilter(cls, input, output, rows, cols, rowWin, colWin):
		halfColWin = int((colWin + 2) / 2)  # 7->4, 8->5
		halfRowWin = int((rowWin + 2) / 2)
		arr = np.array(input, dtype=np.float64)
		arr2d = arr.reshape(-1, rows)  # preserve original k*rows+l stride
		padded = np.zeros((arr2d.shape[0] + 1, arr2d.shape[1] + 1), dtype=np.float64)
		padded[1:, 1:] = arr2d
		integral = padded.cumsum(axis=0).cumsum(axis=1)
		i_idx = np.arange(rows)
		j_idx = np.arange(cols)
		xmin = np.maximum(0, i_idx - halfRowWin)
		xmax = np.minimum(rows, i_idx + halfRowWin)
		ymin = np.maximum(0, j_idx - halfColWin)
		ymax = np.minimum(cols, j_idx + halfColWin)
		xmin_g, ymin_g = np.meshgrid(xmin, ymin, indexing='ij')
		xmax_g, ymax_g = np.meshgrid(xmax, ymax, indexing='ij')
		window_sums = (integral[xmax_g, ymax_g]
		             - integral[xmin_g, ymax_g]
		             - integral[xmax_g, ymin_g]
		             + integral[xmin_g, ymin_g])
		window_areas = (xmax_g - xmin_g) * (ymax_g - ymin_g)
		result = window_sums / window_areas
		flat = result.ravel()
		for k in range(flat.size):
			output[k] = float(flat[k])

	@classmethod
	def prettyHash(cls,hash):
		#Hashes are 16x16. Print in this format
		if len(hash.hash)!=256:
			print("This function only works with 256-bit hashes.")
			return
		return np.array(hash.hash).astype(int).reshape((16,16))
