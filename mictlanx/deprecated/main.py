from mictlanx.v4.client import Client
from mictlanx.v4.interfaces.index import Peer
from mictlanx.utils.segmentation import Chunks
from option import Some
from rory.core.utils.SegmentationUtils import Segmentation
from rory.core.security.dataowner import DataOwner
from rory.core.security.cryptosystem.liu import Liu
from concurrent.futures import ProcessPoolExecutor
# from rory.core.utils.Utils import Utils
# Utils.f
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
r           = 620
a           = 26
plain_m     = np.random.random((r,a))
max_workers = 2
num_chunks  = 4
m           = 3
executor = ProcessPoolExecutor(max_workers=max_workers)
liu = Liu()
dataowner = DataOwner(
        m=m,
        liu_scheme=liu,
        sens=0.01
)

dm = dataowner.get_U(plaintext_matrix=plain_m,algorithm="NNC")
print("DM_SHAPE AND TYPE",dm.shape,dm.dtype)

encrypted_chs = Segmentation.segment_and_encrypt_liu_with_executor(
    executor=executor,
    key="aa",
    plaintext_matrix = plain_m,
    dataowner = dataowner,
    n = r*a*m,
    num_chunks=num_chunks, 
    max_workers=max_workers
)

for chunk in encrypted_chs.iter():
    print(chunk.metadata)

x = encrypted_chs.to_ndarray()
print(x.unwrap()[0].shape)

executor.shutdown()

with open("/rory/rory-client-0/source/audit_data_model.npy","rb") as f:
    plaintext_matrix = np.load(f)
    print(plaintext_matrix.shape,plaintext_matrix.dtype)

# PUT IN MICTLANX
# px= c.put_chunks(
#     key="aa2",
#     chunks=encrypted_chs,
#     tags={},
#     bucket_id="test"
# )

# for p in px:
#     if p.is_ok:
#         print("PUTTED",p.unwrap())

# print(plain_m)
# xs =np.random.random((598,2,4))
# maybe_chunks = Chunks.from_ndarray(ndarray = xs, group_id="BALLID",chunk_prefix=Some("BALLID"),num_chunks=4)
# if maybe_chunks.is_some:
    # chunks = maybe_chunks.unwrap()
    # for chunk in chunks.iter():
        # print(chunk.chunk_id)
    # print(chunks.unwrap())
# res = c.get_and_merge_with_num_chunks(key="dbsnnc1-encrypted-DM",bucket_id="rory",num_chunks=4).result()
# if res.is_ok:
#     x = res.unwrap()
#     print(x.metadata.tags)
#     xs = np.frombuffer(x.value,dtype="float64")
    
    # print(620*26*3)620