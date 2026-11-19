"""The portable classifier format must agree with scikit-learn.

sklearn_portable_classifiers.py reimplements two estimators in numpy, so that
BoxPoseClassifier can evaluate a trained pose classifier without sklearn and
without being pinned to the sklearn version it was trained with. That is the
right trade, but it is a reimplementation: 273 lines that have to produce the
same numbers as the library they replace.

This is the conformance test for that, the counterpart to
test_awa_conformance.py. It needs sklearn, which is a development dependency
(the `classic` extra), so it skips where only the runtime is installed.
"""

import numpy as np
import pytest

from hsdetect.cv.sklearn_portable_classifiers import (model_from_dict,
                                                      read_box_pose_classifier_json,
                                                      write_box_pose_classifier_json)

sklearn = pytest.importorskip('sklearn',
                              reason='sklearn is only needed to WRITE a '
                                     'classifier; install the classic extra')

FEATURES = ['dist_to_centre', 'area', 'aspect_ratio']
FRAME_SIZE = (160, 120)

# Measured agreement: the trees reproduce sklearn exactly, the SVM to ~1e-14.
# yolotiny's POSE_EXPORT_TOLERANCE is 1e-9 for the same comparison.
TOLERANCE = 1e-9


@pytest.fixture(scope='module')
def dataset():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture(scope='module')
def svm_dataset():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 3))
    y = (X[:, 0] - X[:, 2] > 0).astype(int)
    return X, y


def _round_trip(tmp_path, estimator, X, **kwargs):
    """Write the estimator, read it back and return the numpy model."""
    file = tmp_path / 'pose.json'

    write_box_pose_classifier_json(file, estimator,
                                   threshold=0.4,
                                   features=FEATURES,
                                   frame_size=FRAME_SIZE,
                                   **kwargs)

    data = read_box_pose_classifier_json(file)

    return data, model_from_dict(data['model'])


def test_gradient_boosted_trees_agree_with_sklearn(tmp_path, dataset):
    from sklearn.ensemble import HistGradientBoostingClassifier

    X, y = dataset
    estimator = HistGradientBoostingClassifier(max_iter=30,
                                               random_state=0).fit(X, y)

    data, portable = _round_trip(tmp_path, estimator, X,
                                 model_name='HistGradientBoosting')

    assert data['model']['type'] == 'gradient_boosted_trees'
    np.testing.assert_allclose(portable.predict_proba(X),
                               estimator.predict_proba(X),
                               rtol=0, atol=TOLERANCE)


def test_calibrated_rbf_svm_agrees_with_sklearn(tmp_path, svm_dataset):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    X, y = svm_dataset
    estimator = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', CalibratedClassifierCV(SVC(kernel='rbf'), method='sigmoid',
                                       ensemble=False, cv=3)),
    ]).fit(X, y)

    data, portable = _round_trip(tmp_path, estimator, X, model_name='SVM (RBF)')

    assert data['model']['type'] == 'rbf_svm_sigmoid_calibrated'
    np.testing.assert_allclose(portable.predict_proba(X),
                               estimator.predict_proba(X),
                               rtol=0, atol=TOLERANCE)


def test_an_unsupported_calibration_is_refused(tmp_path, svm_dataset):
    """The writer checks what it is given rather than writing a file that
    would evaluate to something else."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    X, y = svm_dataset
    estimator = Pipeline([
        ('scaler', StandardScaler()),
        # ensemble=True averages several calibrators; the portable form has
        # one sigmoid.
        ('clf', CalibratedClassifierCV(SVC(kernel='rbf'), method='sigmoid',
                                       ensemble=True, cv=3)),
    ]).fit(X, y)

    with pytest.raises(ValueError, match='ensemble=False'):
        write_box_pose_classifier_json(tmp_path / 'pose.json', estimator,
                                       threshold=0.4, features=FEATURES,
                                       frame_size=FRAME_SIZE)


def test_the_written_file_records_how_it_was_trained(tmp_path, dataset):
    """The file has to say which sklearn produced it, so a disagreement found
    later can be traced -- the numbers alone do not say that."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    X, y = dataset
    estimator = HistGradientBoostingClassifier(max_iter=5,
                                               random_state=0).fit(X, y)

    data, _ = _round_trip(tmp_path, estimator, X, model_name='HGB')

    assert data['versions']['sklearn'] == sklearn.__version__
    assert data['trained_with']['class'].endswith('HistGradientBoostingClassifier')
    assert data['features'] == FEATURES
    assert data['frame_size'] == list(FRAME_SIZE)


def test_the_committed_classifiers_still_load():
    """The three pose classifiers in demos/ must stay readable: they are what
    the live chains reference."""
    import pathlib

    files = sorted(pathlib.Path('demos/live/multiprocessed/models'
                                '/PoseClassifier').glob('*.json'))

    assert files, 'no committed pose classifiers found'

    for file in files:
        data = read_box_pose_classifier_json(file)
        model = model_from_dict(data['model'])

        X = np.zeros((2, len(data['features'])))
        proba = model.predict_proba(X)

        assert proba.shape == (2, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
