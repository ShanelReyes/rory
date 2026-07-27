import numpy as np
from rory.core.utils.utils import Utils
from rory.core.security.cryptosystem.pqc.ckks import CkksModes



def test_get_threshold():
    dm = np.array([
        [0.0, 3.0, 5.0],
        [3.0, 0.0, 2.0],
        [5.0, 2.0, 0.0],
    ])
    threshold = Utils.get_threshold(distance_matrix=dm)
    assert threshold == 2.0


def test_get_labelvector_from_indexes():
    c_indexes = [[0, 2], [1, 3]]
    labels = Utils.get_labelvector_from_indexes(shape=4, c_indexes=c_indexes)
    assert labels == [0, 1, 0, 1]


def test_get_labelvector_from_indexes_order():
    c_indexes = [[1, 3], [0, 2]]
    labels = Utils.get_labelvector_from_indexes(shape=4, c_indexes=c_indexes)
    assert labels == [1, 0, 1, 0]


def test_generate_centroids():
    matrix = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
    ])
    centroids = Utils.generate_centroids(k=2, plain_matrix=matrix)
    assert centroids.shape == (2, 2)
    assert np.array_equal(centroids[0], [1.0, 2.0])
    assert np.array_equal(centroids[1], [3.0, 4.0])


def test_get_shape_of_matrix_ndarray():
    arr = np.zeros((3, 4, 5))
    shape = Utils.getShapeOfMatrix(arr)
    assert shape == (3, 4, 5)


def test_get_shape_of_matrix_list():
    lst = [[1, 2], [3, 4], [5, 6]]
    shape = Utils.getShapeOfMatrix(lst)
    assert shape == (3, 2)


def test_verify_zero():
    assert bool(Utils.verifyZero(np.zeros((3, 3)))) is True
    assert bool(Utils.verifyZero(np.array([[0, 1], [0, 0]]))) is False


def test_fill_label_vector():
    result = Utils.fillLabelVector(label_vector=[2, 2, 3, 3], k=2)
    assert result[:2] == [0, 1]
    assert result[2:] == [2, 2, 3, 3]


def test_fill_label_vector_empty():
    result = Utils.fillLabelVector(label_vector=[], k=3)
    assert result == [0, 1, 2]


def test_calculate_similarity():
    udm = np.array([
        [[0, 0], [1, 2], [3, 4]],
        [[1, 2], [0, 0], [5, 6]],
        [[3, 4], [5, 6], [0, 0]],
    ], dtype=np.float64)
    sim = Utils.calculateSimilarity(UDM=udm, limit=2, sim=0.0, xy=(0, 1))
    assert sim == 3.0


def test_distance_default():
    result = Utils.distance(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert result.shape == (2, 2, 2)


def test_distance_diff_abs():
    result = Utils.distance(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert result.shape == (2, 2, 2)


def test_empty_cluster():
    clusters = Utils.empty_cluster(3)
    assert len(clusters) == 3
    assert clusters == [[], [], []]


def test_compute_mean_relative_error_identical():
    old = np.array([1.0, 2.0, 3.0])
    new = np.array([1.0, 2.0, 3.0])
    mre = Utils.compute_mean_relative_error(old=old, new=new)
    assert mre == 0.0


def test_compute_mean_relative_error_difference():
    old = np.array([10.0, 20.0, 30.0])
    new = np.array([11.0, 18.0, 33.0])
    mre = Utils.compute_mean_relative_error(old=old, new=new)
    assert mre > 0.0


def test_compute_mean_relative_error_zeros():
    old = np.array([0.0, 0.0])
    new = np.array([0.0, 0.0])
    mre = Utils.compute_mean_relative_error(old=old, new=new, eps=0.001)
    assert mre == 0.0


def test_verify_mean_error_true():
    old = np.array([1.0, 2.0])
    new = np.array([1.0, 2.0])
    assert Utils.verify_mean_error(old_matrix=old, new_matrix=new, min_error=0.15) is True


def test_verify_mean_error_false():
    old = np.array([1.0, 2.0])
    new = np.array([10.0, 20.0])
    assert Utils.verify_mean_error(old_matrix=old, new_matrix=new, min_error=0.15) is False


def test_populate_clusters(liu_scheme):
    sk = liu_scheme.generate_secret_key()
    plaintext = np.array([
        [1.0, 2.0],
        [1.1, 2.1],
        [10.0, 11.0],
        [10.1, 11.1],
    ], dtype=np.float64)
    enc_result = liu_scheme.encryptMatrix(plaintext_matrix=plaintext, secret_key=sk)
    encrypted = enc_result.data

    n = plaintext.shape[0]
    a = plaintext.shape[1]
    udm = np.zeros((n, n, a))
    for i in range(n):
        for j in range(n):
            udm[i, j] = plaintext[i] - plaintext[j]

    initial_clusters = [encrypted[:1].tolist(), encrypted[1:2].tolist()]

    result = Utils.populateClusters(
        record_id=2,
        UDM=udm,
        clusters=initial_clusters,
        ciphertext_matrix=encrypted,
    )
    assert result.is_ok
    clusters, labels = result.unwrap()
    assert len(clusters) >= 2
    assert len(labels) == 2


def test_calculate_centroids(liu_scheme):
    sk = liu_scheme.generate_secret_key()
    m = liu_scheme.m
    plaintext = np.array([
        [1.0, 2.0],
        [1.1, 2.1],
        [10.0, 11.0],
    ], dtype=np.float64)
    enc_result = liu_scheme.encryptMatrix(plaintext_matrix=plaintext, secret_key=sk)
    encrypted = enc_result.data

    clusters = [encrypted[:2].tolist(), encrypted[2:3].tolist()]

    result = Utils.calculateCentroids(
        clusters=clusters,
        k=2,
        attributes=2,
        m=m,
    )
    assert result.is_ok
    centroids = result.unwrap()
    assert isinstance(centroids, np.ndarray)
    assert len(centroids) == 2
    assert len(centroids[0]) == 2
    assert len(centroids[0][0]) == m


def test_compute_centroid_shift_liu(liu_scheme):
    sk = liu_scheme.generate_secret_key()
    m = liu_scheme.m
    plaintext = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    enc_result = liu_scheme.encryptMatrix(plaintext_matrix=plaintext, secret_key=sk)
    encrypted = enc_result.data
    prev = np.array([encrypted[0].tolist(), encrypted[1].tolist()])
    curr = np.array([encrypted[1].tolist(), encrypted[0].tolist()])
    shift = Utils.compute_centroid_shift_liu(
        previous_centroids=prev,
        current_centroids=curr,
    )
    assert isinstance(shift, np.ndarray)
    assert shift.shape == (2, 2, m)


def test_get_scale(ckks_client):
    plaintext = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    ct = ckks_client.encryptVector(plaintext)
    scale = Utils.get_scale(ct)
    assert scale is not None
    assert scale > 0


def test_ptxt_from_scalar(ckks_client):
    he = ckks_client.he_object
    pt = Utils.ptxt_from_scalar(HE=he, val=0.5, n_features=4, scale=int(he.scale))
    assert pt is not None


def test_relinearize_if_possible(ckks_client):
    he = ckks_client.he_object
    plaintext = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    ct = ckks_client.encryptVector(plaintext)
    result = Utils.relinearize_if_possible(HE=he, ct=ct)
    assert result is not None


def test_try_rescale_next(ckks_client):
    he = ckks_client.he_object
    plaintext = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    ct = ckks_client.encryptVector(plaintext)
    result = Utils.try_rescale_next(HE=he, ct=ct)
    assert result is not None


def test_rebind_ct(ckks_client):
    plaintext = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    ct = ckks_client.encryptVector(plaintext)
    copy_ct = Utils.rebind_ct(ct)
    assert copy_ct is not None
    assert id(copy_ct) != id(ct)


def test_align(ckks_client):
    he = ckks_client.he_object
    plaintext = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    ct = ckks_client.encryptVector(plaintext)
    pt = Utils.ptxt_from_scalar(HE=he, val=0.5, n_features=4, scale=int(he.scale))
    a_al, b_al = Utils.align(HE=he, a=ct, b=pt, only_mod=True)
    assert a_al is not None
    assert b_al is not None


def test_normalize_scale(ckks_client):
    he = ckks_client.he_object
    plaintext = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    ct = ckks_client.encryptVector(plaintext)
    scale = int(he.scale)
    normalized = Utils.normalize_scale(HE=he, ct=ct, scale=scale)
    assert normalized is not None


def test_safe_add(ckks_client):
    he = ckks_client.he_object
    plaintext = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    enc_a = ckks_client.encryptVector(plaintext)
    enc_b = ckks_client.encryptVector(plaintext)
    result = Utils.safe_add(HE=he, a=enc_a, b=enc_b)
    assert result is not None


def test_safe_sub(ckks_client):
    he = ckks_client.he_object
    plaintext_a = np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float64)
    plaintext_b = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64)
    enc_a = ckks_client.encryptVector(plaintext_a)
    enc_b = ckks_client.encryptVector(plaintext_b)
    result = Utils.safe_sub(HE=he, a=enc_a, b=enc_b)
    assert result is not None


def test_safe_multiply(ckks_client):
    he = ckks_client.he_object
    scale = ckks_client.SECURITY_LEVELS[CkksModes.ML.value][128]["scale"]
    plaintext = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64)
    enc_a = ckks_client.encryptVector(plaintext)
    enc_b = ckks_client.encryptVector(plaintext)
    result = Utils.safe_multiply(HE=he, a=enc_a, b=enc_b, scale=scale)
    assert result is not None


def test_mul_plain_scalar(ckks_client):
    he = ckks_client.he_object
    scale = ckks_client.SECURITY_LEVELS[CkksModes.ML.value][128]["scale"]
    plaintext = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64)
    enc = ckks_client.encryptVector(plaintext)
    result = Utils.mul_plain_scalar(HE=he, ct=enc, scalar=0.5, scale=scale, n_features=4)
    assert result is not None


def test_add_plain_scalar(ckks_client):
    he = ckks_client.he_object
    scale = ckks_client.SECURITY_LEVELS[CkksModes.ML.value][128]["scale"]
    plaintext = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64)
    enc = ckks_client.encryptVector(plaintext)
    result = Utils.add_plain_scalar(HE=he, ct=enc, scalar=1.0, n_features=4, scale=scale)
    assert result is not None


def test_dot_cipher_garbage(ckks_client):
    he = ckks_client.he_object
    scale = ckks_client.SECURITY_LEVELS[CkksModes.ML.value][128]["scale"]
    n_features = 4
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    w = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    enc_x = ckks_client.encryptVector(x)
    enc_w = ckks_client.encryptVector(w)
    result = Utils.dot_cipher_garbage(
        HE=he, x_ct=enc_x, w_ct=enc_w, n_features=n_features, scale=float(scale)
    )
    assert result is not None


def test_safe_add_all_levels_identical(ckks_client):
    he = ckks_client.he_object
    plaintext = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    enc_a = ckks_client.encryptVector(plaintext)
    enc_b = ckks_client.encryptVector(plaintext)
    result = Utils.safe_add(HE=he, a=enc_a, b=enc_b)
    assert result is not None
    dec = ckks_client.decryptVector(result)
    assert dec is not None


def test_dot_cipher_garbage_simple(ckks_client):
    he = ckks_client.he_object
    scale = ckks_client.SECURITY_LEVELS[CkksModes.ML.value][128]["scale"]
    vec = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    enc = ckks_client.encryptVector(vec)
    result = Utils.dot_cipher_garbage(
        HE=he, x_ct=enc, w_ct=enc, n_features=4, scale=float(scale)
    )
    assert result is not None


def test_get_scale_encrypted_vector(ckks_client):
    plaintext = np.ones(4, dtype=np.float64)
    enc = ckks_client.encryptVector(plaintext)
    ct = enc
    s = Utils.get_scale(ct)
    assert s is not None
    assert isinstance(s, float)
