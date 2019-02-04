import os
from nilearn.datasets.utils import _fetch_files

'''
Example function from nilearn to download data from osf
'''


def fetch_localizer_button_task(data_dir=None, url=None, verbose=1):
    """Fetch left vs right button press contrast maps from the localizer.
    This functison ships only 2nd subject (S02) specific tmap and
    its normalized T1 image.
    Parameters
    ----------
    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location.
    url: string, optional
        Override download URL. Used for test only (or if you setup a mirror of
        the data).
    verbose: int, optional
        verbosity level (0 means no message).
    Returns
    -------
    data: Bunch
        Dictionary-like object, the interest attributes are :
        'tmap': string, giving paths to nifti contrast maps
        'anat': string, giving paths to normalized anatomical image
    Notes
    ------
    This function is only a caller for the fetch_localizer_contrasts in order
    to simplify examples reading and understanding.
    The 'left vs right button press' contrast is used.
    See Also
    ---------
    nilearn.datasets.fetch_localizer_calculation_task
    nilearn.datasets.fetch_localizer_contrasts
    """
    # The URL can be retrieved from the nilearn account on OSF (Open
    # Science Framework). Uploaded files specific to S02 from
    # fetch_localizer_contrasts ['left vs right button press']
    if url is None:
        url = 'https://osf.io/dx9jn/download'

    tmap = "t_map_left_auditory_&_visual_click_vs_right_auditory&visual_click.nii.gz"
    anat = "normalized_T1_anat_defaced.nii.gz"

    opts = {'uncompress': True}

    options = ('tmap', 'anat')
    filenames = [(os.path.join('localizer_button_task', name), url, opts)
                 for name in (tmap, anat)]

    dataset_name = 'brainomics'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    files = _fetch_files(data_dir, filenames, verbose=verbose)

    fdescr = _get_dataset_descr('brainomics_localizer')

    params = dict([('description', fdescr)] + list(zip(options, files)))
    return Bunch(**params)


def fetch_ibc_subjects_contrasts(data_dir=None, url=None, verbose=1):
    """Fetch left vs right button press contrast maps from the localizer.
    This functison ships only 2nd subject (S02) specific tmap and
    its normalized T1 image.
    Parameters
    ----------
    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location.
    url: string, optional
        Override download URL. Used for test only (or if you setup a mirror of
        the data).
    verbose: int, optional
        verbosity level (0 means no message).
    Returns
    -------
    data: Bunch
        Dictionary-like object, the interest attributes are :
        'tmap': string, giving paths to nifti contrast maps
        'anat': string, giving paths to normalized anatomical image
    Notes
    ------
    This function is only a caller for the fetch_localizer_contrasts in order
    to simplify examples reading and understanding.
    The 'left vs right button press' contrast is used.
    See Also
    ---------
    nilearn.datasets.fetch_localizer_calculation_task
    nilearn.datasets.fetch_localizer_contrasts
    """
    # The URL can be retrieved from the nilearn account on OSF (Open
    # Science Framework). Uploaded files specific to S02 from
    # fetch_localizer_contrasts ['left vs right button press']
    if url is None:
        url = 'https://osf.io/dx9jn/download'

    tmap = "t_map_left_auditory_&_visual_click_vs_right_auditory&visual_click.nii.gz"
    anat = "normalized_T1_anat_defaced.nii.gz"

    opts = {'uncompress': True}

    options = ('tmap', 'anat')
    filenames = [(os.path.join('localizer_button_task', name), url, opts)
                 for name in (tmap, anat)]

    dataset_name = 'brainomics'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    files = _fetch_files(data_dir, filenames, verbose=verbose)

    fdescr = _get_dataset_descr('brainomics_localizer')

    params = dict([('description', fdescr)] + list(zip(options, files)))
    return Bunch(**params)
