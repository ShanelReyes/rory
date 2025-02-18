import pytest
from mictlanx.v4.client import Client
from mictlanx.v4.interfaces import Router
from roryclient.client import RoryClient
import time as T
client = RoryClient(hostname="localhost",port=3001)


def test_download():
    client = Client(
        client_id="rory_client_mictlanx",
        bucket_id="rory",
        log_output_path="/rory/log",
        routers=[Router(
            router_id="mictlanx-router-0",
            protocol="http",
            ip_addr="localhost",
            port=60666
        )]
    )
    t1 = T.time()
    res = client.get_with_retry(key="encryptedsknnmodel1bbbbb", chunk_size="50MB")
    print("RES", res)
    print("RESPONSE_TIME", T.time()-t1)

@pytest.mark.skip("")
def test_knn_pqc():
    MAX_REQS= 1
    for i in range(MAX_REQS):
        id                    = "sknn2xxx"
        model_filename        = "classificationc0r10a5k20model"
        model_labels_filename = "classificationc0r10a5k20modellabels"
        model_tests_filename  = "classificationc0r10a5k20data"
        num_chunks            = 2
        extension             = "npy"
        result = client.sknn_pqc_train(
            id              = id,
            model_filename        = model_filename,
            model_labels_filename = model_labels_filename,
            record_tests_filename =model_tests_filename,
            num_chunks            = num_chunks,
            extension             = extension
        )
        print("TRAIN_RESULT",result)


        # if result.is_ok:
        #     res = result.unwrap()
        #     predict_response = client.sknn_pqc_predict(
        #             id              = id,
        #             model_filename        = model_filename,
        #             model_labels_filename = model_labels_filename,
        #             record_tests_filename =model_tests_filename,
        #             num_chunks            = num_chunks,
        #             extension             = extension,
        #             encrypted_model_dtype= res.encrypted_model_dtype,
        #             encrypted_model_shape= res.encrypted_model_shape
        #     )
        #     print("PREDICT_RESULT",predict_response)
