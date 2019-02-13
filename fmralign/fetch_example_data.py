import os
from nilearn.datasets.utils import _fetch_files, _get_dataset_dir

'''
Example function from nilearn to download data from osf
'''


def fetch_ibc_subjects_contrasts(subjects=[1, 2], conditions='all', url='https://osf.io/dx9jn/download', data_dir=None, verbose=1):
    """Fetch all IBC contrast maps for each of subjects.
    After downloading all relevant images that are not already cached, it returns a dataframe with all needed links.

    Parameters
    ----------
    subjects : list of ints <12.
        Subjects data to download
    conditions: str "all" by default, int or list of str
        Pick the conditions to fetch for each subject, all of them by default
        If a list is provided, each str is supposed to be an existing condition to fetch
    url: string, optional
        Override download URL. Used for test only (or if you setup a mirror of
        the data).
    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location.
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
    This function is a caller _fetch_files in order
    to simplify examples reading and understanding for fmralign.
    See Also
    ---------
    nilearn.datasets.fetch_localizer_calculation_task
    nilearn.datasets.fetch_localizer_contrasts
    """
    # The URL can be retrieved from the nilearn account on OSF
    """all_conditions = ["audio_left_button_press", "audio_right_button_press", "video_left_button_press",     "video_right_button_press", "horizontal_checkerboard", "vertical_checkerboard", "audio_sentence", "video_sentence", "audio_computation", "video_computation", "saccades", "rotation_hand", "rotation_side", "object_grasp", "object_orientation", "mechanistic_audio", "mechanistic_video", "triangle_mental", "triangle_random", "false_belief_audio",
                      "false_belief_video", "speech_sound", "non_speech_sound", "face_gender", "face_control", "face_trusty", "expression_intention", "expression_gender", "expression_control", "story", "math", "mental", "random", "punishment", "reward", "left_hand", "right_hand", "left_foot", "right_foot", "tongue", "cue", "shape", "face", "relational", "match", "0back_body", "2back_body", "0back_face", "2back_face", "0back_tools", "2back_tools", "0back_place", "2back_place"]
    subjects = ['sub-%02d' %
                i for i in [1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]]
    """
    opts = {'uncompress': True}
    dataset_name = 'ibc'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)

    df = _fetch_files(data_dir, ("data_parser.csv",
                                 url, opts), verbose=verbose)
    if conditions == 'all':
        conditions = df.contrast.unique()
    elif not isinstance(conditions, list):
        raise TypeError(
            "conditions argument should be 'all', an int or a list of strings")

    df['path'] = df['path'].str.replace('path_to_dir', data_dir)
    df[df.subject.isin(subjects)][
        df.contrast.isin(conditions)].path.values[-1]

    mask = _fetch_files(
        data_dir, ("mask_ibc_gm_3mm.nii.gz", url, opts), verbose=verbose)
    files = []
    for subject in subjects:
        filenames = [(os.path.join(subject, "%s_ap.nii.gz" % condition), url, opts)
                     for condition in conditions]
        filenames.extend([(os.path.join(subject, "%s_pa.nii.gz" % condition), url, opts)
                          for condition in conditions])
        files.append(_fetch_files(data_dir, filenames, verbose=verbose))
    # question should I return subject variable that is useful to filter, subject should be int in df.

    return df, mask
