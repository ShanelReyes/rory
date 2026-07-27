from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from rory.core.security.cryptosystem.liu import Liu
from rory.core.security.cryptosystem.pqc.ckks import Ckks, CkksModes
from rory.core.security.cryptosystem.paillier import Paillier
from rory.core.security.cryptosystem.fdhope import Fdhope


class SchemeParams(ABC):
	@abstractmethod
	def create_scheme(self):
		"""Create and return a configured cryptographic scheme instance.

		Returns:
			HomomorphicCipher or Tuple: The scheme instance or a tuple
				of schemes for composite configurations.
		"""
		...


@dataclass
class LiuParams(SchemeParams):
	security_level: int = 128
	_round: bool = False
	decimals: int = 2
	secure_random: bool = False
	seed: Optional[int] = None
	use_np_random: bool = False
	output_path: str = "/sink"
	save: bool = False

	def create_scheme(self) -> Liu:
		"""Create a Liu cryptosystem instance with the stored parameters.

		Generates keys and optionally persists them to disk.

		Returns:
			Liu: Initialised Liu scheme with generated keys.
		"""
		liu = Liu(
			_round=self._round,
			decimals=self.decimals,
			secure_random=self.secure_random,
			seed=self.seed,
			use_np_random=self.use_np_random,
			security_level=self.security_level,
		)
		liu.generate_keys(security_level=self.security_level)
		if self.save:
			liu.save_keys(self.output_path)
		return liu


@dataclass
class CkksParams(SchemeParams):
	security_level: int = 128
	decimals: int = 2
	mode: CkksModes = CkksModes.DEFAULT
	enable_relinearize: bool = True
	enable_rotate: bool = True
	keys_path: Optional[str] = None
	output_path: str = "/rory/keys"
	save: bool = False
	relinkey_filename: str = ""
	rotatekey_filename: str = ""

	def create_scheme(self) -> Ckks:
		"""Create a CKKS cryptosystem instance with the stored parameters.

		If a keys_path is set, loads existing keys. Otherwise generates
		new keys and optionally saves them.

		Returns:
			Ckks: Initialised CKKS scheme with loaded or generated keys.
		"""
		if self.keys_path is not None:
			return Ckks.from_pyfhel(
				_round             = False,
				decimals           = self.decimals,
				path               = self.keys_path,
				relinkey_filename  = self.relinkey_filename,
				rotatekey_filename = self.rotatekey_filename
			)
		ckks = Ckks()
		ckks.generate_keys(
			mode               = self.mode,
			security_level     = self.security_level,
			decimals           = self.decimals,
			output_path        = self.output_path,
			save               = self.save,
			enable_relinearize = self.enable_relinearize,
			enable_rotate      = self.enable_rotate,
		)
		return ckks


@dataclass
class PaillierParams(SchemeParams):
	security_level: int = 128
	output_path: str = "/sink"
	filename: str = "rory-phe"
	save: bool = False

	def create_scheme(self) -> Paillier:
		"""Create a Paillier cryptosystem instance with generated keys.

		Generates a keypair and optionally persists it to disk.

		Returns:
			Paillier: Initialised Paillier scheme with generated keypair.
		"""
		paillier = Paillier()
		paillier.generate_keys(
			security_level=self.security_level,
			output_path=self.output_path,
			filename=self.filename,
			save=self.save,
		)
		return paillier


@dataclass
class LiuAndFdhopeParams(LiuParams):
	fdhope_seed: int = 42
	fdhope_min_val: float = 0.0
	fdhope_max_range: int = 8
	fdhope_proportion: int = 15
	fdhope_range_limit: int = 2
	fdhope_interval_length: float = 0.001
	sens: float = 0.00001

	def create_scheme(self) -> tuple:
		"""Create a Liu and FDHOPE scheme pair.

		Delegates to LiuParams for the Liu scheme and creates a
		new FDHOPE instance with the configured seed.

		Returns:
			tuple: (Liu, Fdhope) initialised schemes.
		"""
		liu = super().create_scheme()
		fdhope = Fdhope(seed=self.fdhope_seed)
		return liu, fdhope


@dataclass
class CkksAndFdhopeParams(CkksParams):
	fdhope_seed: int = 42
	fdhope_min_val: float = 0.0
	fdhope_max_range: int = 8
	fdhope_proportion: int = 15
	fdhope_range_limit: int = 2
	fdhope_interval_length: float = 0.001
	sens: float = 0.00001

	def create_scheme(self) -> tuple:
		"""Create a CKKS and FDHOPE scheme pair.

		Delegates to CkksParams for the CKKS scheme and creates a
		new FDHOPE instance with the configured seed.

		Returns:
			tuple: (Ckks, Fdhope) initialised schemes.
		"""
		ckks = super().create_scheme()
		fdhope = Fdhope(seed=self.fdhope_seed)
		return ckks, fdhope
