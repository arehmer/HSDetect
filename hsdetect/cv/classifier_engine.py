# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 10:36:43 2024

@author: rehmer
"""

from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import _is_fitted
from sklearn.preprocessing import scale

import typing
import pickle as pkl
from pathlib import Path
import numpy as np
import cv2
import pandas as pd


from .misc import pd_to_cv2_htpa_DataFrame, cv2_to_pd_DataFrame



def build_classifier_engine(**kwargs):
    name = kwargs.pop('name')
    if name == 'ClassEngineLinSVM_CV':
        return ClassEngineLinSVM_CV(**kwargs)
    elif name == 'ClassEngineSVC':
        return ClassEngineSVC(**kwargs)
    elif name == 'RandomForest':
        return RandomForest(**kwargs)

import ast       
def get_classes():
    with open(__file__, 'r') as f:
        tree = ast.parse(f.read(), filename=__file__)
    return [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

class ClassifierEngine():
        
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

# class OpenCV_SVM(classifier_engine):
#     '''
#     Uses the OpenCV implementation of a Kernel Support Vector Machine
#     https://docs.opencv.org/4.x/d1/d2d/classcv_1_1ml_1_1SVM.html
#     '''
    
#     _name = 'OpenCV_SVM'
    
#     def __init__(self,**kwargs):
#         """
#         OpenCV_SVM needs its own initialization function, because OpenCV
#         objects do not posses the __dict__ function to return all class
#         attributes. Instead of assigning all class attributes using 
#         setattr() an entirelly new object has to be created.

#         Parameters
#         ----------
#         **kwargs : TYPE
#             DESCRIPTION.

#         Returns
#         -------
#         None.

#         """
#         # Mapping from readable words for normal humans to the flag-based 
#         # clusterfuck of C++
#         kernel_types = {'linear':cv2.ml.SVM_LINEAR,
#                         'poly':cv2.ml.SVM_POLY,
#                         'rbf':cv2.ml.SVM_RBF,
#                         'sigmoid':cv2.ml.SVM_SIGMOID,
#                         'chi2':cv2.ml.SVM_CHI2,
#                         'inter':cv2.ml.SVM_INTER}
                
#         self.kernel_type = kernel_types[kwargs.pop('kernel_type','rbf')]
        
#         # Mapping from readable words for normal humans to the flag-based 
#         # clusterfuck of C++
#         svm_types = {'C_SVC':cv2.ml.SVM_C_SVC,
#                      'NU_SVC':cv2.ml.SVM_NU_SVC,
#                      'ONE_CLASS':cv2.ml.SVM_ONE_CLASS,
#                      'EPS_SVR':cv2.ml.SVM_EPS_SVR,
#                      'NU_SVR':cv2.ml.SVM_NU_SVR}
        
        
#         self.svm_type = svm_types[kwargs.pop('svm_type','C_SVC')]
        
#         self._svm = self._init_svm(**kwargs)
        
#     def _init_svm(self,**kwargs):
#         """
#         This class generates the svm object and returns it

#         Parameters
#         ----------
#         **kwargs : TYPE
#             DESCRIPTION.

#         Returns
#         -------
#         None.

#         """
        
#         svm = cv2.ml.SVM_create()

#         svm.setType(self.svm_type)
#         svm.setKernel(self.kernel_type)
#         # svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-8))

#         return svm  
    
#     def fit(self,X:np.ndarray,y:np.ndarray):
#         '''
#         This function is called by the user or other functions to fit the 
#         parameters of the SVM.
#         In this case trainAuto() is used for parameter optimization, which
#         automatically performs cross-validation and varies the hyperparameters 
#         to find an optimal setting 

#         Parameters
#         ----------
#         X : np.ndarray
#             DESCRIPTION.
#         y : np.ndarray : TYPE
#             DESCRIPTION.

#         Returns
#         -------
#         None.

#         '''
        
#         # Cast data to the format that OpenCV likes
#         X = X.astype(np.float32)
#         y = y.astype(np.int32).reshape((-1,1))
        
        
#         # Set hyperparameters the same way sklearn does (because opencv svm)
#         # sucks
#         Cgrid = cv2.ml.SVM.getDefaultGridPtr(cv2.ml.SVM_C)
#         Cgrid.minVal = 1
#         Cgrid.maxVal = Cgrid.minVal
        
#         gammaGrid = cv2.ml.SVM.getDefaultGridPtr(cv2.ml.SVM_GAMMA)
#         gammaGrid.minVal = 1/(X.shape[1] * X.var())
#         gammaGrid.maxVal = gammaGrid.minVal
        
#         # Fit the SVM
#         self._svm.trainAuto(samples = X,
#                             layout = cv2.ml.ROW_SAMPLE,
#                             responses = y,
#                             Cgrid = Cgrid,
#                             gammaGrid = gammaGrid)
#         # self._svm.trainAuto(samples = X,
#         #                     layout = cv2.ml.ROW_SAMPLE,
#         #                     responses = y)

#     def score(self,X,y):
#         return 0
    
#     def decision_function(self,X):
        
#         # Cast data to the format that OpenCV likes
#         X = X.astype(np.float32)

#         # Set flag=1 to return the raw output (=distance to hyperplane) instead
#         # of the class label. See here for documentation of flags:
#         # https://docs.opencv.org/4.8.0/db/d7d/classcv_1_1ml_1_1StatModel.html#af1ea864e1c19796e6264ebb3950c0b9a
#         retval,y = self._svm.predict(X,flags=1)  
        
#         return y  
        
#     def save(self) -> dict:
#         """
#         Returns all non-private an non-builtin attributes of this class
#         as a dictionary with the purpose of reloading this instance from the
#         attribute dictionary. 

#         Parameters
#         ----------
#         folder : Path
#             DESCRIPTION.

#         Returns
#         -------
#         dict
#             DESCRIPTION.

#         """
        
#         attr_dict = { k:v for k,v in vars(self).items() if not k.startswith('_') }

#         return attr_dict


        
        