# -*- coding: utf-8 -*-
import os
from nilearn.datasets.utils import _fetch_files, _get_dataset_dir
import pandas as pd


def fetch_ibc_subjects_contrasts(subjects, data_dir=None, verbose=1):
    """Fetch all IBC contrast maps for each of subjects.
    After downloading all relevant images that are not already cached,
    it returns a dataframe with all needed links.

    Parameters
    ----------
    subjects : list of str.
        Subjects data to download. Available strings are ['sub-01', 'sub-02',
        'sub-04' ... 'sub-09', 'sub-11' ... sub-15]
    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location.
    verbose: int, optional
        verbosity level (0 means no message).

    Returns
    -------
    files : list of list of str
        List (for every subject) of list of path (for every conditions),
        in ap then pa acquisition.
    metadata_df : Pandas Dataframe
        Table containing some metadata for each available image in the dataset,
        as well as their pathself.
        Filtered to contain only the 'subjects' parameter metadatas
    mask: str
        Path to the mask to be used on the data
    Notes
    ------
    This function is a caller to nilearn.datasets.utils._fetch_files in order
    to simplify examples reading and understanding for fmralign.
    See Also
    ---------
    nilearn.datasets.fetch_localizer_calculation_task
    nilearn.datasets.fetch_localizer_contrasts
    """
    # The URLs can be retrieved from the nilearn account on OSF
    if subjects is "all":
        subjects = ['sub-%02d' %
                    i for i in [1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]]
    dataset_name = 'ibc'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)

    # download or retrieve metadatas, put it in a dataframe,
    # list all condition and specify path to the right directory
    metadata_path = _fetch_files(data_dir, [('ibc_3mm_all_subjects_metadata.csv',
                                             "https://osf.io/pcvje/download",
                                             {"uncompress": True})],
                                 verbose=verbose)
    metadata_df = pd.read_csv(metadata_path[0])
    conditions = metadata_df.condition.unique()
    metadata_df['path'] = metadata_df['path'].str.replace(
        'path_to_dir', data_dir)
    # filter the dataframe to return only rows relevant for subjects argument
    metadata_df = metadata_df[metadata_df.subject.isin(subjects)]

    # download / retrieve mask niimg and find its path
    mask = _fetch_files(
        data_dir, [('gm_mask_3mm.nii.gz', "https://osf.io/yvju3/download",
                    {"uncompress": True})], verbose=verbose)[0]

    # list all url keys for downloading separetely each subject data
    url_keys = {"sub-01": "8z23h", "sub-02": "e9kbm", "sub-04": "qn5b6",
                "sub-05": "u74a3", "sub-06": "83bje", "sub-07": "43j69",
                "sub-08": "ua8qx", "sub-09": "bxwtv", "sub-11": "3dfbv",
                "sub-12": "uat7d", "sub-13": "p238h", "sub-14": "prdk4",
                "sub-15": "sw72z"}

    # for all subjects in argument, download all contrasts images and list
    # their path in the variable files
    opts = {'uncompress': True}
    files = []
    for subject in subjects:
        url = "https://osf.io/%s/download" % url_keys[subject]
        filenames = [(os.path.join(subject, "%s_ap.nii.gz" % condition),
                      url, opts) for condition in conditions]
        filenames.extend([(os.path.join(subject, "%s_pa.nii.gz" % condition),
                           url, opts) for condition in conditions])
        files.append(_fetch_files(data_dir, filenames, verbose=verbose))
    return files, metadata_df, mask
