# -*- coding: utf-8 -*-
"""
Trained classifiers stored as plain JSON instead of pickled scikit-learn
objects.

A pickled scikit-learn model can only be loaded with the scikit-learn
version it was saved with, because the pickle holds scikit-learn's internal
objects. The classes here hold only the numbers a prediction needs (tree
splits and leaf values, support vectors and coefficients, ...) and compute
predict_proba() with numpy alone. They give the same probabilities as the
scikit-learn model they were exported from.

Supported models (the "type" key of a model dict):

    gradient_boosted_trees      HistGradientBoostingClassifier, binary,
                                numerical features
    rbf_svm_sigmoid_calibrated  Pipeline(StandardScaler,
                                CalibratedClassifierCV(SVC(kernel='rbf'),
                                method='sigmoid', ensemble=False))

Writing a file (write_box_pose_classifier_json, used by the training code)
needs scikit-learn; reading and prediction need only numpy.
"""

import json
from pathlib import Path

import numpy as np


# Format tag of a BoxPoseClassifier JSON file
BOX_POSE_CLASSIFIER_FORMAT = 'hsdetect.box_pose_classifier/1'


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class GradientBoostedTrees:
    """
    Binary gradient boosted trees.

    raw = baseline + sum over all trees of the value of the leaf reached
    P(class 1) = sigmoid(raw)

    In each tree, start at node 0. At node i go to left[i] if
    x[feature[i]] <= threshold[i], otherwise to right[i]; a missing value
    (NaN) goes to the left if missing_left[i]. Stop at a node with
    is_leaf[i].
    """

    type = 'gradient_boosted_trees'

    def __init__(self, classes, n_features, baseline, trees, **kwargs):
        self.classes_ = np.asarray(classes)
        self.n_features_in_ = int(n_features)
        self.baseline = float(baseline)
        self.trees = [{k: np.asarray(v) for k, v in tree.items()}
                      for tree in trees]

    def raw_prediction(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        rows = np.arange(len(X))
        raw = np.full(len(X), self.baseline)

        for t in self.trees:
            # Walk all samples down the tree at once
            node = np.zeros(len(X), dtype=int)
            while True:
                inner = ~t['is_leaf'][node]
                if not inner.any():
                    break
                x = X[rows, t['feature'][node]]
                go_left = np.where(np.isnan(x), t['missing_left'][node],
                                   x <= t['threshold'][node])
                nxt = np.where(go_left, t['left'][node], t['right'][node])
                node = np.where(inner, nxt, node)
            raw += t['value'][node]

        return raw

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = _sigmoid(self.raw_prediction(X))
        return np.column_stack([1.0 - p1, p1])

    def to_dict(self) -> dict:
        return {'type': self.type,
                'description': self.__doc__.strip(),
                'classes': self.classes_.tolist(),
                'n_features': self.n_features_in_,
                'baseline': self.baseline,
                'trees': [{k: v.tolist() for k, v in t.items()}
                          for t in self.trees]}


class RBFSVMSigmoidCalibrated:
    """
    Standardized features, an RBF support vector machine and a sigmoid
    calibration of its decision value.

    z   = (x - scaler_mean) / scaler_scale
    dec = sum_i dual_coef[i] * exp(-gamma * |z - support_vectors[i]|^2)
          + intercept
    P(class 1) = 1 / (1 + exp(calibration_a * dec + calibration_b))
    """

    type = 'rbf_svm_sigmoid_calibrated'

    def __init__(self, classes, scaler_mean, scaler_scale, gamma,
                 support_vectors, dual_coef, intercept, calibration_a,
                 calibration_b, **kwargs):
        self.classes_ = np.asarray(classes)
        self.scaler_mean = np.asarray(scaler_mean, dtype=float)
        self.scaler_scale = np.asarray(scaler_scale, dtype=float)
        self.gamma = float(gamma)
        self.support_vectors = np.asarray(support_vectors, dtype=float)
        self.dual_coef = np.asarray(dual_coef, dtype=float)
        self.intercept = float(intercept)
        self.calibration_a = float(calibration_a)
        self.calibration_b = float(calibration_b)
        self.n_features_in_ = self.support_vectors.shape[1]

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, dtype=float) - self.scaler_mean) / self.scaler_scale
        dec = np.empty(len(Z))
        for i in range(0, len(Z), 64):            # in chunks to limit memory
            d2 = ((Z[i:i+64, None, :] - self.support_vectors[None, :, :])**2).sum(-1)
            dec[i:i+64] = np.exp(-self.gamma * d2) @ self.dual_coef
        return dec + self.intercept

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        dec = self.decision_function(X)
        p1 = _sigmoid(-(self.calibration_a * dec + self.calibration_b))
        return np.column_stack([1.0 - p1, p1])

    def to_dict(self) -> dict:
        return {'type': self.type,
                'description': self.__doc__.strip(),
                'classes': self.classes_.tolist(),
                'scaler_mean': self.scaler_mean.tolist(),
                'scaler_scale': self.scaler_scale.tolist(),
                'gamma': self.gamma,
                'intercept': self.intercept,
                'calibration_a': self.calibration_a,
                'calibration_b': self.calibration_b,
                'dual_coef': self.dual_coef.tolist(),
                'support_vectors': self.support_vectors.tolist()}


_MODEL_TYPES = {cls.type: cls for cls in (GradientBoostedTrees,
                                          RBFSVMSigmoidCalibrated)}


def model_from_dict(d: dict):
    """Model dict (the "model" entry of a classifier JSON) -> model object."""
    try:
        cls = _MODEL_TYPES[d['type']]
    except KeyError:
        raise ValueError(f"Unknown model type {d.get('type')!r}; known: "
                         f"{list(_MODEL_TYPES)}") from None
    return cls(**{k: v for k, v in d.items() if k not in ('type', 'description')})


# ---------------------------------------------------------------------------
# Export from scikit-learn, for the training code
# ---------------------------------------------------------------------------

def model_to_dict(est) -> dict:
    """Fitted scikit-learn estimator -> model dict, see module docstring."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import SVC

    if isinstance(est, HistGradientBoostingClassifier):
        if len(est.classes_) != 2 or est.n_trees_per_iteration_ != 1:
            raise ValueError('Only binary HistGradientBoostingClassifiers are supported.')
        trees = []
        for (predictor,) in est._predictors:
            n = predictor.nodes
            if n['is_categorical'].any():
                raise ValueError('Categorical splits are not supported.')
            trees.append({'feature': n['feature_idx'].astype(int),
                          'threshold': n['num_threshold'].astype(float),
                          'missing_left': n['missing_go_to_left'].astype(bool),
                          'left': n['left'].astype(int),
                          'right': n['right'].astype(int),
                          'is_leaf': n['is_leaf'].astype(bool),
                          'value': n['value'].astype(float)})
        return GradientBoostedTrees(
            classes=est.classes_, n_features=est.n_features_in_,
            baseline=np.asarray(est._baseline_prediction).ravel()[0],
            trees=trees).to_dict()

    if isinstance(est, Pipeline) and len(est.steps) == 2 and \
        isinstance(est.steps[0][1], StandardScaler) and \
        isinstance(est.steps[1][1], CalibratedClassifierCV):

        scaler, calibrated = est.steps[0][1], est.steps[1][1]
        if len(calibrated.calibrated_classifiers_) != 1 or \
            calibrated.calibrated_classifiers_[0].method != 'sigmoid':
            raise ValueError('Only CalibratedClassifierCV with ensemble=False '
                             'and method="sigmoid" is supported.')
        cc = calibrated.calibrated_classifiers_[0]
        svc = cc.estimator
        if not (isinstance(svc, SVC) and svc.kernel == 'rbf') or \
            len(svc.classes_) != 2:
            raise ValueError('Only a binary SVC with kernel="rbf" is supported.')
        if not np.array_equal(svc.classes_, calibrated.classes_):
            raise ValueError('Calibrator and SVC disagree on the classes.')

        mean = scaler.mean_ if scaler.with_mean else np.zeros(scaler.n_features_in_)
        scale = scaler.scale_ if scaler.with_std else np.ones(scaler.n_features_in_)
        return RBFSVMSigmoidCalibrated(
            classes=calibrated.classes_, scaler_mean=mean, scaler_scale=scale,
            gamma=svc._gamma, support_vectors=svc.support_vectors_,
            dual_coef=svc.dual_coef_[0], intercept=svc.intercept_[0],
            calibration_a=cc.calibrators[0].a_,
            calibration_b=cc.calibrators[0].b_).to_dict()

    raise TypeError(f'Cannot export {type(est).__name__} to JSON.')


def write_box_pose_classifier_json(file, estimator, threshold: float,
                                   features: list, frame_size: tuple,
                                   positive_label: str = 'lying',
                                   model_name: str = '',
                                   metadata: dict = None) -> None:
    """
    Write a trained scikit-learn classifier and its settings as a
    BoxPoseClassifier JSON file, e.g. at the end of the training code.

    estimator      : fitted scikit-learn model, see model_to_dict()
    threshold      : P(positive) >= threshold means positive_label
    features       : feature names, in the order the model was trained with
    frame_size     : (width, height) of the frames the boxes come from
    metadata       : optional, e.g. scores and training sequences
    """
    import platform
    import sklearn
    from hsdetect.engine_io import json_dumps

    params = estimator.get_params(deep=False)
    data = {'format': BOX_POSE_CLASSIFIER_FORMAT,
            'name': 'BoxPoseClassifier',
            'model_name': model_name,
            'features': [str(f) for f in features],
            'frame_size': [int(v) for v in frame_size],
            'positive_label': positive_label,
            'threshold': float(threshold),
            'versions': {'python': platform.python_version(),
                         'numpy': np.__version__,
                         'sklearn': sklearn.__version__},
            'metadata': metadata or {},
            'trained_with': {
                'class': f'{type(estimator).__module__}.{type(estimator).__qualname__}',
                'params': {k: v for k, v in params.items()
                           if v is None or isinstance(v, (bool, int, float, str))}},
            'model': model_to_dict(estimator)}

    Path(file).write_text(json_dumps(data) + '\n', encoding='utf-8')


def read_box_pose_classifier_json(file) -> dict:
    """Read a BoxPoseClassifier JSON file."""
    with open(file, 'r', encoding='utf-8') as io:
        data = json.load(io)
    if data.get('format') != BOX_POSE_CLASSIFIER_FORMAT:
        raise ValueError(f"{file} is not a BoxPoseClassifier JSON "
                         f"(format {data.get('format')!r}, expected "
                         f"{BOX_POSE_CLASSIFIER_FORMAT!r}).")
    return data
