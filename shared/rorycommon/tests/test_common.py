import pytest
from rorycommon import Common
from Pyfhel import Pyfhel



async def test_from_pyctxts_to_chunks(ckks):
    x = ckks.encrypt_list([1])
    assert len(x) == 1
    result = Common.from_pyctxts_to_chunks(
        key="test_key",
        xs=x,
        num_chunks=1
    )
    assert result.is_some, "Expected result to be Some, but got None"
    # print(result)