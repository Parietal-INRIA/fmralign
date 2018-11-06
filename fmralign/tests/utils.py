from sklearn.utils.testing import assert_array_almost_equal
from nilearn.input_data import NiftiMasker


def _test_algo(algo, X, Y, mask=None):
    algo.fit(X, Y)
    imtest = algo.transform(X)
    masker = NiftiMasker(mask_img=mask)
    masker.fit()
    array_truth = masker.transform(Y)
    array_test = masker.transform(imtest)
    assert_array_almost_equal(array_truth, array_test, decimal=6)
