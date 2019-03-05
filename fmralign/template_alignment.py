from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from joblib import Parallel, delayed
from sklearn.externals.joblib import Memory
from nilearn.image import index_img
from nilearn.input_data.masker_validation import check_embedded_nifti_masker
from fmralign.pairwise_alignment import PairwiseAlignment


def euclidian_mean(imgs, masker, scale_template=False):
    """ Make the euclidian average of images.

    Parameters
    ----------
    imgs: list of Niimgs
        Each img is 3D by default, but can also be 4D.
    masker:

    scale_template: boolean
        If true, the returned average is scaled to have the average norm of imgs
        If false, it will usually have a smaller norm than initial average
        because noise will cancel across images

    Returns
    -------
    average_img: Niimg
        Average of imgs, with same shape as one img
    """
    masked_imgs = [masker.transform(img) for img in imgs]
    average_img = np.mean(masked_imgs, axis=0)
    scale = 1
    if scale_template:
        X_norm = 0
        for img in masked_imgs:
            X_norm += np.linalg.norm(img)
        X_norm /= len(masked_imgs)
        scale = X_norm / np.linalg.norm(average_img)
    average_img *= scale

    return masker.inverse_transform(average_img)


def _align_images_to_template(imgs, template, alignment_method,
                              n_pieces, clustering_method, n_bags, masker,
                              memory, memory_level, n_jobs, verbose):
    '''Convenience function : for a list of images, return the list
    of estimators (PairwiseAlignment instances) aligning each of them to a
    common target, the template. All arguments are used in PairwiseAlignment
    '''
    aligned_imgs = []
    for img in imgs:
        piecewise_estimator = \
            PairwiseAlignment(n_pieces=n_pieces,
                              alignment_method=alignment_method,
                              clustering_method=clustering_method, n_bags=n_bags,
                              mask=masker, memory=memory,
                              memory_level=memory_level,
                              n_jobs=n_jobs, verbose=verbose)
        piecewise_estimator.fit(img, template)
        aligned_imgs.append(piecewise_estimator.transform(img))
    return aligned_imgs


def create_template(imgs, n_iter, scale_template, alignment_method, n_pieces,
                    clustering_method, n_bags, masker, memory, memory_level,
                    n_jobs, verbose):
    '''Create template through alternate minimization.  Compute iteratively :
        * T minimizing sum(||R_iX_i-T||) which is the mean of aligned images (RX_i)
        * align initial images to new template T
            (find transform R minimizing ||R_iX_i-T|| for each img X_i)


        Parameters
        ----------
        alignment_method: string
            Algorithm used to perform alignment between X_i and Y_i :
            * either 'identity', 'scaled_orthogonal', 'ridge_cv',
                'permutation', 'diagonal'
            * or an instance of one of alignment classes
                (imported from functional_alignment.alignment_methods)
        n_pieces: int, optional (default = 1)
            Number of regions in which the data is parcellated for alignment
            If 1 the alignment is done on full scale data.
            If >1, the voxels are clustered and alignment is performed
                on each cluster applied to X and Y.
        clustering_method : string, optional (default : k_means)
            'k_means' or 'ward', method used for clustering of voxels
        n_bags: int, optional (default = 1)
            If 1 : one estimator is fitted.
            If >1 number of bagged parcellations and estimators used.
        mask: Niimg-like object, instance of NiftiMasker or
                                MultiNiftiMasker, optional (default : None)
            Mask to be used on data. If an instance of masker is passed,
            then its mask will be used. If no mask is given,
            it will be computed automatically by a MultiNiftiMasker
            with default parameters.
        memory: instance of joblib.Memory or string (default : None)
            Used to cache the masking process and results of algorithms.
            By default, no caching is done. If a string is given, it is the
            path to the caching directory.
        memory_level: integer, optional (default : None)
            Rough estimator of the amount of memory used by caching.
            Higher value means more memory for caching.
        n_jobs: integer, optional (default = 1)
            The number of CPUs to use to do the computation. -1 means
            'all CPUs', -2 'all CPUs but one', and so on.
        verbose: integer, optional (default = 0)
            Indicate the level of verbosity. By default, nothing is printed.
    '''

    aligned_imgs = imgs
    template_history = []
    iter = 0
    while True:
        template = euclidian_mean(
            aligned_imgs, masker, scale_template)
        if 0 < iter < n_iter:
            template_history.append(template)
        if iter == n_iter:
            break
        aligned_imgs = _align_images_to_template(imgs, template,
                                                 alignment_method, n_pieces,
                                                 clustering_method, n_bags,
                                                 masker, memory, memory_level,
                                                 n_jobs, verbose)
        iter += 1

    return template, template_history


def map_template_to_image(img, train_index, template, alignment_method,
                          n_pieces, clustering_method, n_bags, masker,
                          memory, memory_level, n_jobs, verbose):
    '''From a template, and new images, learn their alignment mapping
    !!! mapping.fit seem to have wrong argument : if error it's upstream in functional_alignment.template
    - Be sure that alignment_methods are the same in class docs and pairwise_alignment
    # Check everywhere whether template is list of 3D Niimgs or 4D Niimgs.


    Parameters
    ----------
    img: list of 3D Niimgs

    train_index: list of int
        Matching index between imgs and the corresponding template images to use
        to learn alignment. len(train_index) must be equal to len(imgs)
    template: list of 3D Niimgs
        Learnt in a first step now used to learn the mapping
    alignment_method: string
        Algorithm used to perform alignment between X_i and Y_i :
        * either 'identity', 'scaled_orthogonal', 'ridge_cv',
            'permutation', 'diagonal'
        * or an instance of one of alignment classes
            (imported from functional_alignment.alignment_methods)
    n_pieces: int, optional (default = 1)
        Number of regions in which the data is parcellated for alignment
        If 1 the alignment is done on full scale data.
        If >1, the voxels are clustered and alignment is performed
            on each cluster applied to X and Y.
    clustering_method : string, optional (default : k_means)
        'k_means' or 'ward', method used for clustering of voxels
    n_bags: int, optional (default = 1)
        If 1 : one estimator is fitted.
        If >1 number of bagged parcellations and estimators used.
    mask: Niimg-like object, instance of NiftiMasker or
                            MultiNiftiMasker, optional (default : None)
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is given,
        it will be computed automatically by a MultiNiftiMasker
        with default parameters.
    n_jobs: integer, optional (default = 1)
        The number of CPUs to use to do the computation. -1 means
        'all CPUs', -2 'all CPUs but one', and so on.
    verbose: integer, optional (default = 0)
        Indicate the level of verbosity. By default, nothing is printed.

    Returns
    -------
    mapping: instance of PairwiseAlignment class
        Alignment estimator fitted to align the template with the input images'''
    mapping_image = index_img(template, train_index)
    mapping = PairwiseAlignment(n_pieces=n_pieces,
                                alignment_method=alignment_method,
                                clustering_method=clustering_method,
                                n_bags=n_bags, mask=masker, memory=memory,
                                memory_level=memory_level,
                                n_jobs=n_jobs, verbose=verbose)
    mapping.fit(mapping_image, img)
    return mapping


def predict_from_template_and_mapping(template, test_index,  mapping):
    """ From a template, and an alignment estimator, predict new contrasts

    Parameters
    ----------
    template: list of 3D Niimgs
        Learnt in a first step now used to predict some new data
    test_index:
        Index of the images not used to learn the alignment mapping and so
        predictable without overfitting
    mapping: instance of PairwiseAlignment class
        Alignment estimator that should be already fitted

    Returns
    -------
    transformed_image: list of Niimgs
        Prediction corresponding to each template image with inex in test_index
        once realigned to the new subjects
    """
    image_to_transform = index_img(template, test_index)
    transformed_image = mapping.transform(image_to_transform)
    return transformed_image


class TemplateAlignment(BaseEstimator, TransformerMixin):
    """
    Decompose the source and target images into source and target regions
     Use alignment algorithms to align source and target regions independantly.
    """

    def __init__(self, alignment_method="identity", n_pieces=1,
                 clustering_method='k_means', n_bags=1,
                 mask=None, smoothing_fwhm=None, standardize=None,
                 detrend=None, target_affine=None, target_shape=None,
                 low_pass=None, high_pass=None, t_r=None,
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, verbose=0):
        '''
        Parameters
        ----------
        alignment_method: string
            Algorithm used to perform alignment between X_i and Y_i :
            * either 'identity', 'scaled_orthogonal', 'ridge_cv',
                'permutation', 'diagonal'
            * or an instance of one of alignment classes
                (imported from functional_alignment.alignment_methods)
        n_pieces: int, optional (default = 1)
            Number of regions in which the data is parcellated for alignment
            If 1 the alignment is done on full scale data.
            If >1, the voxels are clustered and alignment is performed
                on each cluster applied to X and Y.
        clustering_method : string, optional (default : k_means)
            'k_means' or 'ward', method used for clustering of voxels
        n_bags: int, optional (default = 1)
            If 1 : one estimator is fitted.
            If >1 number of bagged parcellations and estimators used.
        mask: Niimg-like object, instance of NiftiMasker or
                                MultiNiftiMasker, optional (default : None)
            Mask to be used on data. If an instance of masker is passed,
            then its mask will be used. If no mask is given,
            it will be computed automatically by a MultiNiftiMasker
            with defaultparameters.
        smoothing_fwhm: float, optional (default : None)
            If smoothing_fwhm is not None, it gives the size in millimeters
            of the spatial smoothing to apply to the signal.
        standardize : boolean, optional (default : None)
            If standardize is True, the time-series are centered and normed:
            their variance is put to 1 in the time dimension.
        detrend : boolean, optional (default : None)
            This parameter is passed to nilearn.signal.clean.
            Please see the related documentation for details
        target_affine: 3x3 or 4x4 matrix, optional (default : None)
            This parameter is passed to nilearn.image.resample_img.
            Please see the related documentation for details.
        target_shape: 3-tuple of integers, optional (default : None)
            This parameter is passed to nilearn.image.resample_img.
            Please see the related documentation for details.
        low_pass: None or float, optional (default : None)
            This parameter is passed to nilearn.signal.clean.
            Please see the related documentation for details.
        high_pass: None or float, optional (default : None)
            This parameter is passed to nilearn.signal.clean.
            Please see the related documentation for details.
        t_r: float, optional (default : None)
            This parameter is passed to nilearn.signal.clean.
            Please see the related documentation for details.
        memory: instance of joblib.Memory or string (default : None)
            Used to cache the masking process and results of algorithms.
            By default, no caching is done. If a string is given, it is the
            path to the caching directory.
        memory_level: integer, optional (default : None)
            Rough estimator of the amount of memory used by caching.
            Higher value means more memory for caching.
        n_jobs: integer, optional (default = 1)
            The number of CPUs to use to do the computation. -1 means
            'all CPUs', -2 'all CPUs but one', and so on.
        verbose: integer, optional (default = 0)
            Indicate the level of verbosity. By default, nothing is printed.
        '''
        self.template = None
        self.template_history = None
        self.alignment_method = alignment_method
        self.n_pieces = n_pieces
        self.clustering_method = clustering_method
        self.n_bags = n_bags
        self.mask = mask
        self.smoothing_fwhm = smoothing_fwhm
        self.standardize = standardize
        self.detrend = detrend
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.memory = memory
        self.memory_level = memory_level
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, imgs, scale_template=False, n_iter=2, save_template=None):
        """
        Learn a template from imgs

        Parameters
        ----------
        imgs: List of Niimg-like objects
           See http://nilearn.github.io/manipulating_images/input_output.html
           source data. Every img must have the same length (number of sample)
        scale_template: boolean, default False
            rescale template after each inference so that it keeps
            the same norm as the average of training images.
        n_iter: int
           number of iteration in the alternate minimization. Each img is
           aligned n_iter times to the evolving template. If n_iter = 0,
           the template is simply the mean of the input images.
        save_template: None or string(optional)
            If not None, path to which the template will be saved.

        Returns
        -------
            self
        TODO : test save_template


        """
        self.masker_ = check_embedded_nifti_masker(self)
        self.masker_.n_jobs = 1  # self.n_jobs
        # Avoid warning with imgs != None
        # if masker_ has been provided a mask_img
        if self.masker_.mask_img is None:
            self.masker_.fit(imgs)
        else:
            self.masker_.fit()

        self.template, self.template_history = \
            create_template(imgs, n_iter, scale_template,
                            self.alignment_method, self.n_pieces,
                            self.clustering_method, self.n_bags,
                            self.masker_, self.memory, self.memory_level,
                            self.n_jobs, self.verbose)
        if save_template is not None:
            self.template.to_filename(save_template)

    def transform(self, imgs, train_index, test_index):
        """
        Predict data from X

        Parameters
        ----------
        imgs: List of Niimg-like objects
           See http://nilearn.github.io/manipulating_images/input_output.html
           source data. Every img must have length (number of sample)
           train_index.
        train_index : list of ints
            indexes of the 3D samples used to map each img to the template
        test_index : list of ints
            indexes of the 3D samples to predict from the template and the mapping

        Returns
        -------
        predicted_imgs: Niimg-like object
           See http://nilearn.github.io/manipulating_images/input_output.html
           predicted data
           List of 4D images, each of them has the same length as the list test_index

        """

        fitted_mappings = Parallel(self.n_jobs, backend="threading",
                                   verbose=self.verbose)(
            delayed(map_template_to_image)
            (img, train_index, self.template, self.alignment_method,
             self.n_pieces, self.clustering_method, self.n_bags, self.masker_,
             self.memory, self.memory_level, self.n_jobs, self.verbose
             ) for img in imgs
        )

        predicted_imgs = Parallel(self.n_jobs, backend="threading",
                                  verbose=self.verbose)(
            delayed(predict_from_template_and_mapping)
            (self.template, test_index, mapping) for mapping in fitted_mappings
        )
        return predicted_imgs
