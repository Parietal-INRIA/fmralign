from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from joblib import Parallel, delayed
from sklearn.externals.joblib import Memory
from nilearn.image import mean_img, index_img
from fmralign.pairwise_alignment import PairwiseAlignment


def euclidian_mean(imgs, scale_template, masker):
    '''imgs is a list of 3D or 4D images
    each img is 3D by default:'''
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

    return masker.inverse_transform(mean_img)


def align_images_to_template(imgs, template, method_alignment, n_pieces, n_bags, masker, n_iter, n_jobs, reg):
    '''
    - Still should take care of clustering method/joint arguments
    - Be sure that alignment_methods are the same in class docs and pairwise_alignment
    - Additional arguments are available in pairwise_alignment : perturbation=False, smoothing_fwhm=None, standardize=None, detrend=False, target_affine=None, target_shape=None, low_pass=None, high_pass=None, t_r=None, memory=Memory(cachedir=None), memory_level=0.

    same for align_template_to_images
    '''
    aligned_imgs = []
    for img in imgs:
        piecewise_estimator = PairwiseAlignment(n_pieces=n_pieces, alignment_method=method_alignment,
                                                mask=masker, clustering_method='k_means', joint_clustering=False, n_bags=n_bags, n_jobs=n_jobs)
        piecewise_estimator.fit(img, template)
        aligned_imgs.append(piecewise_estimator.transform(img))
    return aligned_imgs


def create_template(imgs, method_alignment, n_pieces, n_bags, scale_template, masker, n_iter, n_jobs, reg):
    '''create_template by deforming images s.t. :, at each iteration :
        compute : T minimizing sum(||RX-T||) which is the mean(RX)
        compute new aligned images through transform R minimizing ||RX-T|| for each img
    '''
    aligned_imgs = imgs
    template_history = []
    iter = 0
    while True:
        template = euclidian_mean(
            aligned_imgs, scale_template, masker)
        if 0 < iter < n_iter:
            template_history.append(template)
        if iter == n_iter:
            break
        aligned_imgs = align_images_to_template(
            imgs, template, method_alignment, n_pieces, n_bags, masker, n_iter, n_jobs, reg)
        iter += 1

    return template, template_history


def map_template_to_image(img, train_index, template, n_pieces, method_alignment, n_bags, masker, n_jobs, reg):
    '''
    !!! mapping.fit seem to have wrong argument : if error it's upstream in functional_alignment.template
    - Still should take care of clustering method/joint arguments
    - Be sure that alignment_methods are the same in class docs and pairwise_alignment
    - Additional arguments are available in pairwise_alignment : perturbation=False, smoothing_fwhm=None, standardize=None, detrend=False, target_affine=None, target_shape=None, low_pass=None, high_pass=None, t_r=None, memory=Memory(cachedir=None), memory_level=0.

    same for align_template_to_images, align_images_to_template
    '''

    mapping_image = index_img(img, train_index)
    mapping = PairwiseAlignment(n_pieces=n_pieces, alignment_method=method_alignment, mask=masker,
                                clustering_method='k_means', joint_clustering=False, n_bags=n_bags, n_jobs=n_jobs)
    mapping.fit(mapping_image, img)
    return mapping


def predict_from_template_and_mapping(template, test_index,  mapping):
    image_to_transform = index_img(template, test_index)
    transformed_image = mapping.transform(image_to_transform)
    return transformed_image


class TemplateAlignment(BaseEstimator, TransformerMixin):
    """
    Decompose the source and target images into source and target regions
     Use alignment algorithms to align source and target regions independantly.
    """

    def __init__(self, n_pieces=1, alignment_method="mean", n_bags=1, reg=1,
                 mask=None,
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1,
                 verbose=0):
        self.template = None
        self.template_history = None
        self.n_pieces = n_pieces
        self.alignment_method = alignment_method
        self.n_bags = n_bags
        self.mask = mask
        self.memory = memory
        self.memory_level = memory_level
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.reg = reg

    def fit(self, imgs, scale_template=False, n_iter=2):
        """
        Learn a template from imgs

        Parameters
        ----------
        imgs: List of Niimg-like objects
           See http://nilearn.github.io/manipulating_images/input_output.html
           source data. Every img must have the same length (number of sample)
        scale_template: boolean, default False
            rescale template after each inference so that it keeps the same norm has the average of training images.
        n_iter: int
           number of iteration in the alternate minimization. Each img is aligned n_iter times to the evolving template. If n_iter = 0, the template is simply the mean of the input images.
        Returns
        -------
            self
        """
        self.template, self.template_history = create_template(
            imgs, self.alignment_method, self.n_pieces, self.n_bags, scale_template, self.mask, n_iter, self.n_jobs, self.reg)

    def transform(self, imgs, train_index, test_index):
        """
        Predict data from X
        Parameters
        ----------
        imgs: List of Niimg-like objects
           See http://nilearn.github.io/manipulating_images/input_output.html
           source data. Every img must have the same length (number of sample) as imgs used in the fit()
           and as the template.
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

        fitted_mappings = Parallel(self.n_jobs, backend="threading", verbose=self.verbose)(
            delayed(map_template_to_image)(
                img, train_index, self.template, self.n_pieces, self.alignment_method, self.n_bags, self.mask, self.n_jobs, self.reg
            ) for img in imgs
        )

        predicted_imgs = Parallel(self.n_jobs, backend="threading", verbose=self.verbose)(
            delayed(predict_from_template_and_mapping)(self.template, test_index, mapping
                                                       ) for mapping in fitted_mappings
        )
        return predicted_imgs
