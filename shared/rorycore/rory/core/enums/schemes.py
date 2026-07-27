from enum import Enum


class Scheme(Enum):
	NONE            = "none"
	LIU             = "liu"
	CKKS            = "ckks"
	PAILLIER        = "paillier"
	LIU_AND_FDHOPE  = "liu_and_fdhope"
	CKKS_AND_FDHOPE = "ckks_and_fdhope"
