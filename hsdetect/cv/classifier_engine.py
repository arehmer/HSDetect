# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 10:36:43 2024

@author: rehmer
"""

from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import _is_fitted, has_fit_parameter
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
import sklearn

import importlib
import platform
import typing
import pickle as pkl
from pathlib import Path
import numpy as np
import cv2
import pandas as pd


from .misc import pd_to_cv2_htpa_DataFrame, cv2_to_pd_DataFrame

from hsdetect import Engine

# Marker identifying this .py as a module containing engines
IS_ENGINE_MODULE = True

def build_classifier_engine(**kwargs):
    name = kwargs.pop('name')
    if name == 'ClassEngineLinSVM_CV':
        return ClassEngineLinSVM_CV(**kwargs)
    elif name == 'ClassEngineSVC':
        return ClassEngineSVC(**kwargs)
    elif name == 'RandomForest':
        return RandomForest(**kwargs)
    elif name == 'BoxPoseClassifier':
        return BoxPoseClassifier(**kwargs)
    else:
        print(f"Classifier {name} not known!")

import ast       
def get_classes():
    with open(__file__, 'r') as f:
        tree = ast.parse(f.read(), filename=__file__)
    return [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

class ClassifierEngine(Engine):
        
    def save(self):
        """
        Overrides standard save method, which only stores attributes, but 
        is not capable of storing objects which have a save() method of 
        their own.

        Returns
        -------
        None.

        """
        
        attr_dict = {}
        
        for attr_name in dir(self):
            # get the attribute *on the class*, not the instance
            attr = getattr(self.__class__, attr_name, None)
        
            # check if it's a property
            if isinstance(attr, property):
                
                # Retrieve the value of the property attr
                attr_val = getattr(self, attr_name)
                
                # If the property has a save() method, save the return value of
                # that method
                if hasattr(attr_val, "save") and callable(attr_val.save):
                    attr_dict[attr_name] = attr_val.save()
                else:
                    attr_dict[attr_name] = attr_val
                
        # Append name
        attr_dict['name'] = self.name
                
        return attr_dict
    
    # def save(self):
    #     '''
    #     Returns all non-private an non-builtin attributes of this class
    #     as a dictionary with the purpose of reloading this instance from the
    #     attribute dictionary. 

    #     Returns
    #     -------
    #     None.

    #     '''
        
    #     # Get all instance attributes 
    #     attr_dict = { k:v for k,v in vars(self).items() \
    #                  if not k.startswith('_') }
        
    #     # Throw out the svm_cpp as it cannot be pickled but will
    #     # be initialized when calling self.__init__()
    #     if 'cv_classifier_engine' in self.__dict__:
    #         del attr_dict['cv_classifier_engine']
            
    #     # Throw out the svm_py, which should not be pickled and is initialized
    #     # by the attributes passed
    #     if 'svm_py' in self.__dict__:
    #         del attr_dict['svm_py']
        
    #     # Get all class attributes
    #     class_dict = {}
    #     for attribute in self.__class__.__dict__.keys():
    #         # Check if it's a built-in type
    #         if attribute[:2] != '__':
    #             # Check if its a method
    #             value = getattr(self.__class__, attribute)
    #             if not callable(value):
    #                 # If not append to dict
    #                 class_dict[attribute] = value

    #     # Concatenate both
    #     attr_dict.update(class_dict)
        
    #     return attr_dict
    

class ClassEngineLinSVM_CV(ClassifierEngine):
    
    name = 'ClassEngineLinSVM_CV'
    
    def __init__(self,**kwargs):
        """
        This class generates the svm object and returns it

        Parameters
        ----------
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # Attributes relevant for cpp instance (inference)
        self.decision_boundary = kwargs.pop('decision_boundary',0)
        self.n_class = kwargs.pop('n_class',2)
        self.n_feature = kwargs.pop('n_feature',0)
        self.coef = kwargs.pop('coef',np.array([]))
        self.intercept = kwargs.pop('intercept',np.array([]))
        self.label = kwargs.pop('label',np.array([0,1]))
        
        # Attributes relevant for python instance (training)
        self.dual = kwargs.pop('dual','auto')
        self.max_iter = kwargs.pop('max_iter',10000)
        self.class_weight = kwargs.pop('class_weight','balanced')
        self.C = kwargs.pop('C',1)
        self.verbose = kwargs.pop('verbose',0)
        self.classes = kwargs.pop('classes',np.array([]))
        self.n_features_in_ = kwargs.pop('n_features_in_',0)
        
                
        # In any case initialize the python svm 
        self.initialize_py_instance(**kwargs)
        
        # Check if the python svm is fitted, if so, initialize the Cpp 
        # implementation using these parameters
        if _is_fitted(self.svm_py) == True:
            self.initialize_cpp_instance()
            
    
        return None
    
    @property
    def decision_boundary(self):
        return self._decision_boundary   
    @decision_boundary.setter
    def decision_boundary(self,decision_boundary:float):
        self._decision_boundary = float(decision_boundary)  
        
    @property
    def n_class(self):
        return self._n_class    
    @n_class.setter
    def n_class(self,n_class:int):
        self._n_class = int(n_class)

    @property
    def n_feature(self):
        return self._n_feature   
    @n_feature.setter
    def n_feature(self,n_feature:int):
        self._n_feature = int(n_feature)

    @property
    def coef(self):
        return self._coef    
    @coef.setter
    def coef(self,coef:np.ndarray):
        self._coef = np.asarray(coef,dtype=np.float32).flatten()

    @property
    def intercept(self):
        return self._intercept  
    @intercept.setter
    def intercept(self,intercept:np.ndarray):
        self._intercept = np.asarray(intercept,dtype=float).flatten()  
    
    @property
    def label(self):
        return self._label  
    @label.setter
    def label(self,label:np.ndarray):
        self._label = np.asarray(label,dtype=int).flatten()        

    @property
    def dual(self):
        return self._dual
    @dual.setter
    def dual(self,dual:str):
        self._dual = str(dual)

    @property
    def max_iter(self):
        return self._max_iter  
    @max_iter.setter
    def max_iter(self,max_iter:int):
        self._max_iter = int(max_iter)

    @property
    def class_weight(self):
        return self._class_weight
    @class_weight.setter
    def class_weight(self,class_weight:np.ndarray|str):
        if isinstance(class_weight, np.ndarray):
            self._class_weight = np.asarray(class_weight,dtype=np.float32).flatten()   
        elif isinstance(class_weight, str):
            self._class_weight = str(class_weight)
        else:
            raise TypeError('class_weight must be np.ndarray or str')
 
    @property
    def C(self):
        return self._C
    @C.setter
    def C(self,C:float):
        self._C = float(C)   

    @property
    def verbose(self):
        return self._verbose
    @verbose.setter
    def verbose(self,verbose:int):
        self._verbose = int(verbose)        
        
    @property
    def classes(self):
        return self._classes
    @classes.setter
    def classes(self,classes:np.ndarray):
        self._classes = np.asarray(classes,dtype=int).flatten()  
        
    @property
    def n_features_in_(self):
        return self._n_features_in_
    @n_features_in_.setter
    def n_features_in_(self,n_features_in_:int):
        self._n_features_in_ = int(n_features_in_) 
        

    def initialize_py_instance(self,**kwargs):
        
        new_svm_py = LinearSVC(dual=self.dual,
                               max_iter = self.max_iter,
                               class_weight = self.class_weight,
                               C = self.C,
                               verbose = self.verbose)
        
        if len(self.coef) != 0:
            new_svm_py.coef_ = self.coef
            new_svm_py.intercept_ = self.intercept
            new_svm_py.classes_ = self.classes
            new_svm_py.n_features_in_ = self.n_features_in_

        self.svm_py =  kwargs.pop('svm_py',new_svm_py)

    def initialize_cpp_instance(self):
        # Initialize a Cpp svm with the same parameters as the fitted 
        # svm_py
        self.cv_classifier_engine = cv2.htpa.ClassEngLinSVM(self.decision_boundary,
                                                            self.n_class,
                                                            self.n_feature,
                                                            self.coef,
                                                            self.intercept.item(),
                                                            self.label)

    
    def fit(self,X,y, w = None):
        
        if w is None:
            w = np.ones(y.shape)
        
        # Fit the python svm
        self.svm_py.fit(X,y,w)
        
        # Set attributes needed for Cpp instance initialization
        self.n_class = len(self.svm_py.classes_.flatten())
        self.n_feature = self.svm_py.n_features_in_
        self.coef = self.svm_py.coef_.flatten().astype(np.float32)
        self.intercept = self.svm_py.intercept_.flatten().astype(np.float32)
        
        # Initialize a Cpp svm with the same parameters
        self.initialize_cpp_instance()
        
        return self.svm_py.score(X,y)

    def predict(self,
                X:np.ndarray,
                bboxes : pd.Series | pd.DataFrame | cv2.htpa.DataFrame):
        
        # Cast X to proper datatype
        X = X.astype(np.float32)
        
        # Check type of annotation and cast if necessary
        if not isinstance(bboxes, cv2.htpa.DataFrame):
            bboxes_cpp = pd_to_cv2_htpa_DataFrame(bboxes)
        else:
            bboxes_cpp = bboxes
                    
        # Apply decision function
        bboxes_pred = self.decision_function(X,
                                             bboxes_cpp)
        
        return bboxes_pred
        
        
        
        
    def decision_function(self,
                          X : np.ndarray,
                          bboxes : pd.Series | pd.DataFrame | cv2.htpa.DataFrame):
        
        # Cast X to proper datatype
        X = X.astype(np.float32)
        
        # Check type of annotation and cast if necessary
        if not isinstance(bboxes, cv2.htpa.DataFrame):
            bboxes_cpp = pd_to_cv2_htpa_DataFrame(bboxes)
        else:
            bboxes_cpp = bboxes
        
        # If already fitted, use the Cpp implementation to predict label and scores
        if _is_fitted(self.svm_py) == True:
            bboxes_pred_cpp  = self.cv_classifier_engine.process(X,bboxes_cpp)
            bboxes_pred = cv2_to_pd_DataFrame(bboxes_pred_cpp)
            return bboxes_pred
        else:
            return None
        

class RandomForest(ClassifierEngine):
    
    name = 'RandomForest'
    
    def __init__(self,**kwargs):
        """
        This class generates the classifier engine and returns it

        Parameters
        ----------
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        # Initialize the python instance 
        self.initialize_py_instance(**kwargs)
        
    def initialize_py_instance(self,**kwargs):
        
        new_svm_py = RandomForestClassifier(max_depth = 2)
        
        # if len(self.coef) != 0:
        #     new_svm_py.coef_ = self.coef
        #     new_svm_py.intercept_ = self.intercept
        #     new_svm_py.classes_ = self.classes
        #     new_svm_py.n_features_in_ = self.n_features_in_

        self.classifier_py =  kwargs.pop('classifier_py',new_svm_py)
        
    def predict(self,
                X:np.ndarray,
                bboxes : pd.Series | pd.DataFrame | cv2.htpa.DataFrame):
        
        # Cast X to proper datatype
        X = X.astype(np.float32)
        
        # Check type of annotation and cast if necessary
        if not isinstance(bboxes, cv2.htpa.DataFrame):
            bboxes_cpp = pd_to_cv2_htpa_DataFrame(bboxes)
        else:
            bboxes_cpp = bboxes
                    
        # Apply decision function
        bboxes_pred = self.decision_function(X,
                                             bboxes_cpp)
        
        return bboxes_pred
        
    def decision_function(self,
                          X : np.ndarray,
                          bboxes : pd.Series | pd.DataFrame | cv2.htpa.DataFrame):
        
        # Cast X to proper datatype
        X = X.astype(np.float32)
        
        # Apply random forest
        logprob = self.classifier_py.predict_log_proba(X).flatten()
        
        # Convert to probability [0,1]
        prob = np.exp(logprob)
        
        # Subtract 0.5 to obtain a score that is negative if prob<0.5
        score = prob - 0.5
        
        bboxes_pred = cv2_to_pd_DataFrame(bboxes)
        
        bboxes_pred['score'] = score
        
        # # Check type of annotation and cast if necessary
        # if not isinstance(bboxes, cv2.htpa.DataFrame):
        #     bboxes_cpp = pd_to_cv2_htpa_DataFrame(bboxes)
        # else:
        #     bboxes_cpp = bboxes
        
        # # If already fitted, use the Cpp implementation to predict label and scores
        # if _is_fitted(self.svm_py) == True:
        #     bboxes_pred_cpp  = self.cv_classifier_engine.process(X,bboxes_cpp)
        #     bboxes_pred = cv2_to_pd_DataFrame(bboxes_pred_cpp)
        #     return bboxes_pred
        # else:
        #     return None
        
        return bboxes_pred
        

             
    def fit(self,X,y, w = None):
        
        if w is None:
            w = np.ones(y.shape)
        
        # Fit the python svm
        self.classifier_py.fit(X,y,w)
        
        
        return self.classifier_py.score(X,y)

    
  
class ClassEngineSVC(ClassEngineLinSVM_CV):
    
    name = 'ClassEngineSVC'
    
    def __init__(self,**kwargs):
        """
        This class generates the svm object and returns it

        Parameters
        ----------
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # Attributes relevant for cpp instance (inference)
        self.decision_boundary = kwargs.pop('decision_boundary',0)
        self.n_class = kwargs.pop('n_class',2)
        self.n_feature = kwargs.pop('n_feature',0)
        self.coef = kwargs.pop('coef',np.array([]))
        self.intercept = kwargs.pop('intercept',np.array([]))
        self.label = kwargs.pop('label',np.array([0,1]))
        
        # Attributes relevant for python instance (training)
        self.dual = kwargs.pop('dual','auto')
        self.max_iter = kwargs.pop('max_iter',5000)
        self.class_weight = kwargs.pop('class_weight','balanced')
        self.C = kwargs.pop('C',1)
        self.verbose = kwargs.pop('verbose',0)
        self.classes = kwargs.pop('classes',np.array([]))
        self.n_features_in_ = kwargs.pop('n_features_in_',0)
        
                
        # In any case initialize the python svm 
        self.initialize_py_instance(**kwargs)
        
            
    
    def initialize_py_instance(self,**kwargs):
        
        new_svm_py = SVC()
        
        # if len(self.coef) != 0:
        #     new_svm_py.coef_ = self.coef
        #     new_svm_py.intercept_ = self.intercept
        #     new_svm_py.classes_ = self.classes
        #     new_svm_py.n_features_in_ = self.n_features_in_

        self.svm_py =  kwargs.pop('svm_py',new_svm_py)
        
        
    def predict(self,
                X:np.ndarray,
                bboxes : pd.Series | pd.DataFrame | cv2.htpa.DataFrame):
        
        bboxes_pred = bboxes.copy()
        
        # Cast X to proper datatype
        X = X.astype(np.float32)
        
                    
        # Apply decision function
        score = self.svm_py.decision_function(X)
        
        if score <= 0:
            label = 0
        else:
            label = 1
        
        bboxes_pred.score = score
        bboxes_pred.label = label
        
        # Make pandas dataframe for compatibility
        bboxes_pred = bboxes_pred.to_frame().T
        
        return bboxes_pred
             
    def fit(self,X,y, w = None):
        
        if w is None:
            w = np.ones(y.shape)
        
        # Fit the python svm
        self.svm_py.fit(X,y,w)
        
        # # Set attributes needed for Cpp instance initialization
        # self.n_class = len(self.svm_py.classes_.flatten())
        # self.n_feature = self.svm_py.n_features_in_
        # self.coef = self.svm_py.coef_.flatten().astype(np.float32)
        # self.intercept = self.svm_py.intercept_.flatten().astype(np.float32)
        
        # # Initialize a Cpp svm with the same parameters
        # self.initialize_cpp_instance()
        
        return self.svm_py.score(X,y)

# ---------------------------------------------------------------------------
# Helpers for BoxPoseClassifier
# ---------------------------------------------------------------------------

# Conversion between fitted scikit-learn estimators and dictionaries:
#     {"class": "<module>.<ClassName>",
#      "params": {hyperparameters, i.e. get_params(deep=False)},
#      "fitted": {learned attributes}}
# with nested estimators wrapped as {"__estimator__": {...}}.
# This is the same format as estimator_to_dict() in the training code
# (pose_estimation/func.py). Keep both copies in sync if it ever changes.

_ESTIMATOR_KEY = "__estimator__"


def _to_plain(obj):
    if isinstance(obj, BaseEstimator):
        return {_ESTIMATOR_KEY: estimator_to_dict(obj)}
    if isinstance(obj, list):
        return [_to_plain(o) for o in obj]
    if isinstance(obj, tuple):
        return tuple(_to_plain(o) for o in obj)
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    return obj


def _from_plain(obj):
    if isinstance(obj, dict):
        if set(obj) == {_ESTIMATOR_KEY}:
            return estimator_from_dict(obj[_ESTIMATOR_KEY])
        return {k: _from_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_plain(o) for o in obj]
    if isinstance(obj, tuple):
        return tuple(_from_plain(o) for o in obj)
    return obj


def estimator_to_dict(est: BaseEstimator) -> dict:
    """Fitted estimator -> {"class", "params", "fitted"}."""
    cls = type(est)
    params = est.get_params(deep=False)
    state = est.__getstate__()
    fitted = {k: v for k, v in state.items() if k not in params}
    return {"class": f"{cls.__module__}.{cls.__qualname__}",
            "params": _to_plain(params),
            "fitted": _to_plain(fitted)}


def estimator_from_dict(d: dict) -> BaseEstimator:
    """{"class", "params", "fitted"} -> fitted estimator. Needs the same
    scikit-learn version it was saved with; scikit-learn warns otherwise."""
    module_name, _, class_name = d["class"].rpartition(".")
    cls = getattr(importlib.import_module(module_name), class_name)
    est = cls(**_from_plain(d["params"]))
    est.__setstate__(_from_plain(d["fitted"]))
    return est


def _versions() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "sklearn": sklearn.__version__}


def compute_box_features(bboxes: pd.DataFrame, frame_size: tuple) -> pd.DataFrame:
    """
    Box geometry features, computed exactly as add_features() in the training
    code (pose_estimation/func.py). If features are added or changed there,
    mirror the change here, or the engine will see different inputs than the
    model was trained on.

    Needs columns xtl, ytl, xbr, ybr; w and h are used if present, otherwise
    computed as xbr - xtl and ybr - ytl.
    """
    b = bboxes
    xtl, ytl = b["xtl"].astype(float), b["ytl"].astype(float)
    xbr, ybr = b["xbr"].astype(float), b["ybr"].astype(float)
    w = b["w"].astype(float) if "w" in b else xbr - xtl
    h = b["h"].astype(float) if "h" in b else ybr - ytl
    frame_w, frame_h = frame_size

    long_side = np.maximum(w, h)
    short_side = np.minimum(w, h)

    # Offset of the box center from the frame center; cos2phi = cos(2*phi)
    # of its direction, 0 at the center itself (see add_features in func.py)
    dx = (xtl + xbr) / 2 - frame_w / 2
    dy = (ytl + ybr) / 2 - frame_h / 2
    r2 = dx**2 + dy**2
    cos2phi = np.where(r2 > 0, (dx**2 - dy**2) / r2.where(r2 > 0, 1), 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_log = np.log(w / h)
        return pd.DataFrame({
            "w": w,
            "h": h,
            "d": np.hypot(dx, dy),
            "ratio": w / h,
            "ratio_log": ratio_log,
            "ratio_norm": long_side / short_side,
            "ratio_normlog": np.log(long_side / short_side),
            "area": long_side * short_side,
            "cos2phi": cos2phi,
            "radial_align": ratio_log * cos2phi,
        }, index=b.index)


BOX_FEATURES = ("d", "ratio", "ratio_log", "ratio_norm", "ratio_normlog", "area",
                "cos2phi", "radial_align")

class BoxPoseClassifier(ClassifierEngine):
    """
    Classifies person bounding boxes as lying / non-lying from box geometry
    alone (distance to frame center, area, aspect ratio). The features are
    computed from the boxes themselves, so the X argument of predict() and
    decision_function() is ignored.

    Built only from the dictionary exported by 03_classify.py:

        with open('box_pose_classifier.pkl', 'rb') as f:
            engine = build_classifier_engine(**pkl.load(f))

    save() returns a dictionary of the same shape.

    Results are added as new columns; existing columns, such as a detection
    score or label, are left untouched:
        pose_score : P(lying) - threshold, >= 0 means lying
        pose_label : 1 = lying, 0 = non-lying,
                     -1 = box with zero width or height (not classified)

    Inference runs in Python only; the stored model needs the scikit-learn
    version it was trained with (see the versions property).
    """

    name = 'BoxPoseClassifier'

    _REQUIRED = ('estimator', 'threshold', 'features', 'frame_size')

    def __init__(self, **kwargs):
        missing = [k for k in self._REQUIRED if k not in kwargs]
        if missing:
            raise KeyError(f"BoxPoseClassifier is missing {missing}. Build it from "
                           f"the dictionary exported by 03_classify.py.")

        self.estimator = kwargs.pop('estimator')
        self.threshold = kwargs.pop('threshold')
        self.features = kwargs.pop('features')
        self.frame_size = kwargs.pop('frame_size')
        self.positive_label = kwargs.pop('positive_label', 'lying')
        self.model_name = kwargs.pop('model_name', '')
        self.versions = kwargs.pop('versions', {})
        self.metadata = kwargs.pop('metadata', {})

        self.initialize_py_instance(**kwargs)

    @property
    def estimator(self):
        return self._estimator
    @estimator.setter
    def estimator(self, estimator: dict):
        if not (isinstance(estimator, dict) and {'class', 'params', 'fitted'} <= set(estimator)):
            raise TypeError("estimator must be a dict with keys 'class', 'params', 'fitted'")
        self._estimator = estimator

    @property
    def threshold(self):
        return self._threshold
    @threshold.setter
    def threshold(self, threshold: float):
        self._threshold = float(threshold)

    @property
    def features(self):
        return self._features
    @features.setter
    def features(self, features: list):
        features = [str(f) for f in features]
        unknown = [f for f in features if f not in BOX_FEATURES]
        if unknown:
            raise ValueError(f"Unknown features {unknown}; available: {list(BOX_FEATURES)}")
        self._features = features

    @property
    def frame_size(self):
        return self._frame_size
    @frame_size.setter
    def frame_size(self, frame_size: tuple):
        w, h = frame_size
        self._frame_size = (int(w), int(h))

    @property
    def positive_label(self):
        return self._positive_label
    @positive_label.setter
    def positive_label(self, positive_label: str):
        self._positive_label = str(positive_label)

    @property
    def model_name(self):
        return self._model_name
    @model_name.setter
    def model_name(self, model_name: str):
        self._model_name = str(model_name)

    @property
    def versions(self):
        return self._versions
    @versions.setter
    def versions(self, versions: dict):
        self._versions = dict(versions)

    @property
    def metadata(self):
        return self._metadata
    @metadata.setter
    def metadata(self, metadata: dict):
        self._metadata = dict(metadata)
        
    def __call__(self, bboxes : pd.DataFrame):
        
        return self.predict(None, bboxes)

    def initialize_py_instance(self, **kwargs):
        # Rebuild the trained scikit-learn model from its dictionary.
        # Not a property, so save() stores the dictionary, not this object.
        self.classifier_py = estimator_from_dict(self.estimator)

        n_in = getattr(self.classifier_py, 'n_features_in_', None)
        if n_in is not None and n_in != len(self.features):
            raise ValueError(f"Stored model expects {n_in} features, "
                             f"but features lists {len(self.features)}: {self.features}")

    def _positive_index(self) -> int:
        # The training script encodes lying as 1, non-lying as 0
        return int(np.flatnonzero(self.classifier_py.classes_ == 1)[0])

    @staticmethod
    def _to_pandas(bboxes) -> pd.DataFrame:
        if isinstance(bboxes, pd.Series):        # a single box
            return bboxes.to_frame().T
        if isinstance(bboxes, pd.DataFrame):
            return bboxes.copy()
        return cv2_to_pd_DataFrame(bboxes)       # cv2.htpa.DataFrame

    def features_from_bboxes(self, bboxes) -> tuple:
        """
        Returns (X, valid): the feature matrix in the order of self.features,
        and a boolean mask of boxes with nonzero width and height. Rows of
        X for invalid boxes contain inf/NaN and must not be passed to the model.
        """
        feats = compute_box_features(self._to_pandas(bboxes), self.frame_size)
        valid = ((feats["w"] > 0) & (feats["h"] > 0)).to_numpy()
        return feats[self.features].to_numpy(dtype=float), valid

    def predict(self,
                X: np.ndarray,
                bboxes: pd.Series | pd.DataFrame | cv2.htpa.DataFrame):
        """X is ignored; features are computed from bboxes."""
        return self.decision_function(X, bboxes)

    def decision_function(self,
                          X: np.ndarray,
                          bboxes: pd.Series | pd.DataFrame | cv2.htpa.DataFrame):
        """X is ignored; features are computed from bboxes."""
        bboxes_pred = self._to_pandas(bboxes)
        X_box, valid = self.features_from_bboxes(bboxes_pred)

        score = np.full(len(bboxes_pred), np.nan)
        label = np.full(len(bboxes_pred), -1, dtype=int)
        if valid.any():
            prob = self.classifier_py.predict_proba(X_box[valid])[:, self._positive_index()]
            score[valid] = prob - self.threshold
            label[valid] = (prob >= self.threshold).astype(int)

        bboxes_pred['pose_score'] = score
        bboxes_pred['pose_label'] = label

        return bboxes_pred

    def fit(self, X, y, w=None):
        """
        Refits the model with its stored hyperparameters. The stored threshold
        is kept, since choosing a new one needs separate validation data.

        X : features in the order of self.features, e.g. from
            features_from_bboxes() (drop rows where valid is False)
        y : 1 / 0, or pose strings compared against positive_label
        w : optional sample weights

        Returns the balanced accuracy on (X, y) at the stored threshold.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if y.dtype.kind in 'OUS':
            y = (y == self.positive_label).astype(int)

        model = clone(self.classifier_py)
        final = model.steps[-1][1] if isinstance(model, Pipeline) else model
        if w is not None and not has_fit_parameter(final, "sample_weight"):
            raise ValueError(f"{type(final).__name__} does not support sample weights; call fit() without w")

        if w is None:
            model.fit(X, y)
        elif isinstance(model, Pipeline):
            model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": w})
        else:
            model.fit(X, y, sample_weight=w)

        # Keep the saved dictionary in step with the refitted model
        self.classifier_py = model
        self.estimator = estimator_to_dict(model)
        self.versions = _versions()

        prob = model.predict_proba(X)[:, self._positive_index()]
        return balanced_accuracy_score(y, (prob >= self.threshold).astype(int))