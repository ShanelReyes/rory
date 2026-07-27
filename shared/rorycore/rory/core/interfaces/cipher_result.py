class CipherResult:
	"""Result of a cipher scheme operation.

	Encapsulates encrypted/decrypted data produced by encryption,
	decryption, or homomorphic operations.

	Args:
		data: The resulting data (encrypted or plaintext). Can be a scalar,
			ndarray, list of ciphertexts, or any scheme-specific type.
	"""
	def __init__(self, data):
		self.data = data
