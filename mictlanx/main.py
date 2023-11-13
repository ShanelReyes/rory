from mictlanx.v4.client import Client
from mictlanx.v4.interfaces.index import Peer
from mictlanx.utils.segmentation import Chunks
from option import Some
import numpy as np

c = Client(
    client_id="client-0-rory",
    bucket_id="rory",
    peers= [
        Peer(peer_id="mictlanx-peer-0",ip_addr="localhost", port=7000),
        Peer(peer_id="mictlanx-peer-1",ip_addr="localhost", port=7001),
    ],
    debug=True,
    show_metrics=False,
    daemon=True,
    max_workers=2,
    lb_algorithm="2CHOICES_UF",
)

np.random.seed(323)
xs =np.random.random((598,2,4))
maybe_chunks = Chunks.from_ndarray(ndarray = xs, group_id="BALLID",chunk_prefix=Some("BALLID"),num_chunks=4)
if maybe_chunks.is_some:
    chunks = maybe_chunks.unwrap()
    for chunk in chunks.iter():
        print(chunk.chunk_id)
    # print(chunks.unwrap())
# res = c.get_and_merge_with_num_chunks(key="skmeans-test-xdyh11-encrypted-UDM",bucket_id="rory",num_chunks=4).result()
# if res.is_ok:
#     x = res.unwrap()
#     xs = np.frombuffer(x.value,dtype="float64")
#     print(xs.shape)
