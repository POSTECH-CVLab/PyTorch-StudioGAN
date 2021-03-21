import numpy as np

from make_semi_supervised_dataset import osss_subset, ss_subset


def test_osss_subset():
    N = 100000
    M = 10
    sM = 5
    ratio = 0.2
    labels = np.random.choice(list(range(M)), size=N)
    res = osss_subset(labels, ratio=ratio, subset_class=sM)
    assert len(np.unique(res)) == sM + 1
    for i in range(sM):
        assert np.abs((res == i).sum() - (N/M)*ratio) < (N/M)*ratio/M
    assert np.abs((res == -1).sum() - N*((2-ratio)*(sM/M))) < N*((2-ratio)*(sM/M))/M


def test_ss_subset():
    N = 100000
    M = 10
    ratio = 0.2
    labels = np.random.choice(list(range(M)), size=N)
    res = ss_subset(labels, ratio=ratio)
    assert len(np.unique(res)) == M + 1
    for i in range(M):
        assert np.abs((res == i).sum() - (N/M)*ratio) < (N/M)*ratio/M
    assert np.abs((res == -1).sum() - N*(1-ratio)) < (N*(1-ratio))/M
