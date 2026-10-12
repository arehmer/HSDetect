"""
Created on Tue Nov 21 14:15:11 2023

@author: rehmer
"""
import numpy as np
import cv2
import pandas as pd
from typing import List
import matplotlib.pyplot as plt

from hsdetect.annot.annot import RoI
from . import misc

from hsdetect import Engine

# Marker identifying this .py as a module containing engines
IS_ENGINE_MODULE = True


def build_tracker(**kwargs):
    name = kwargs.pop('name')
    if name == 'BlockTracker':
        return BlockTracker(**kwargs)
    else:
        raise Exception('Tracker ' + name + ' not known!')

import ast       
def get_classes():
    with open(__file__, 'r') as f:
        tree = ast.parse(f.read(), filename=__file__)
    return [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

class Tracker(Engine):
    
    def __init__(self,
                 SensorType:int,
                 **kwargs:dict):
        """
            
        Parameters
        ----------
        width : int
            width of the frame, i.e. width of the sensor array.
        height : int
            height of the frame, i.e. height of the sensor array.

    
        Returns
        -------
        None.
    
        """
        
        
        self.SensorType = SensorType
                
    @property
    def SensorType(self):
        return self._SensorType
    
    @SensorType.setter
    def SensorType(self,SensorType:int):
        self._SensorType = SensorType

    def track(self,
              frame:np.ndarray,
              roi_annot: pd.DataFrame | pd.Series | cv2.htpa.DataFrame):
    
        # Check if inputs have proper type
        if not isinstance(frame,np.ndarray):
            raise TypeError('frame is not type nd.array')

        if not (isinstance(roi_annot,pd.DataFrame) or
                isinstance(roi_annot,pd.Series) or
                isinstance(roi_annot,cv2.htpa.df_row)):
            raise TypeError('roi has wrong type')

        if not frame.dtype == np.uint16:
            raise TypeError('frame is not type uint16')
            
        # Convert roi from pd.Series to cv.htpa.df_row
        if not isinstance(roi_annot, cv2.htpa.DataFrame):
            roi_annot_cv = misc.pd_to_cv2_htpa_row_detect(roi_annot)
        else:
            roi_annot_cv = roi_annot

        # Pass to Cpp instance and retrieve result
        roi_tracked_cv = self.cv_tracker.track(frame,
                                               roi_annot_cv)
        
        # Convert result back to python datatype
        roi_tracked = misc.cv2_row_detect_to_pd_Series(roi_tracked_cv)
        
        roi_tracked = pd.DataFrame(roi_tracked).transpose()
        
        return roi_tracked
    
    def save(self)->dict:
        """
        Standard save method, which only stores properties.

        Returns
        -------
        dict.

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


class BlockTracker(Tracker):
    
    name = 'BlockTracker'
    
    def __init__(self,**attr_dict):
        
        # Call constructor of parent class
        SensorType = attr_dict.pop('SensorType',None)
        super(BlockTracker, self).__init__(SensorType,
                                           **attr_dict)
        
        self.pix_shift = attr_dict.pop('pix_shift',(1,))  
        self.MotionModel = attr_dict.pop('MotionModel',cv2.htpa.Unconstrained)  
        
        # At this point, the only kernel implemented/tested is Epanechnikov
        # Therefore hardcode this kernel at this point
        self.kernel = misc.Kernel['Epanechnikov']
        
        # Initialize OpenCV instance of that class
        self.initialize_cpp_instance()
        
    @property
    def pix_shift(self):
        return self._pix_shift
    
    @pix_shift.setter
    def pix_shift(self,pix_shift: tuple):
        
        # Check input has proper type
        if not isinstance(pix_shift,tuple):
            raise TypeError('pix_shift must be <tuple>')
        
        self._pix_shift = pix_shift 
        
        # if isinstance(pix_shift,list):
        #     if not all([type(p) is int for p in pix_shift]):
        #         raise TypeError('pix_shift must be of type int')
        #     # Assign value to property
        #     self._pix_shift = np.array(pix_shift)
        
        # if isinstance(pix_shift,np.ndarray):
        #     if not np.issubdtype(pix_shift.dtype, np.integer):
        #         raise TypeError('pix_shift must be of type int')
        #     # Assign value to property
        #     self._pix_shift = pix_shift 
    

    @property
    def kernel(self):
        return self._kernel
    
    @kernel.setter
    def kernel(self,kernel:int):
        
        # Assign value to property
        self._kernel = kernel       
        
    @property
    def MotionModel(self):
        return self._MotionModel
    
    @MotionModel.setter
    def MotionModel(self,MotionModel:int):
        
        # Assign value to property
        self._MotionModel = MotionModel
         
    def initialize_cpp_instance(self):
        
        self.cv_tracker = cv2.htpa.BlockTracker(self.SensorType,
                                                self.pix_shift,
                                                self.kernel,
                                                self.MotionModel)
        
    def init_tracking_target(self,
                             frame:np.ndarray,
                             roi_annot: pd.DataFrame | cv2.htpa.DataFrame):
    
        # Check if inputs have proper type
        if not isinstance(frame,np.ndarray):
            raise TypeError('frame is not type nd.array')

        if not (isinstance(roi_annot,pd.DataFrame) or
                isinstance(roi_annot,cv2.htpa.df_row)):
            raise TypeError('roi has wrong type')

        if not frame.dtype == np.uint16:
            raise TypeError('frame is not type uint16')
            
        # Convert roi from pd.Series to cv.htpa.df_row
        if not isinstance(roi_annot, cv2.htpa.DataFrame):
            roi_annot_cv = misc.pd_to_cv2_htpa_row_detect(roi_annot)
        else:
            roi_annot_cv = roi_annot
        
        self.cv_tracker.init_tracking_target(frame,
                                             roi_annot_cv)
    
# class TemplateTracker(Tracker):
    
#     def __init__(self,
#                  pyramid_levels  : int = 3,
#                  pyramid_scale :float = 0.9,
#                  n_pix_search : int = 3,
#                  template_type : str = 'appearance'):
        
#         # The number of pyramid levels needs to be an uneven number
#         if pyramid_levels % 2 != 1:
#             raise ValueError('Number of pyramid levels needs to be uneven!')
            
        
#         self.pyramid_levels = pyramid_levels
#         self.pyramid_scale = pyramid_scale
        
#         self.roi_template = None
#         self.template_type = template_type
#         self.n_pix_search = n_pix_search
        

#     @property
#     def pyramid_levels(self):
#         return self._pyramid_levels
#     @pyramid_levels.setter
#     def pyramid_levels(self,n_levels):
#         self._pyramid_levels = n_levels
        
#     @property
#     def pyramid_scale(self):
#         return self._pyramid_scale
#     @pyramid_scale.setter
#     def pyramid_scale(self,s):
#         self._pyramid_scale = s
        
#     @property
#     def n_pix_search(self):
#         return self._n_pix_search
#     @n_pix_search.setter
#     def n_pix_search(self,n_pix):
#         self._n_pix_search = n_pix
        
#     @property
#     def template_type(self):
#         return self._template_type
#     @template_type.setter
#     def template_type(self,template_type):
#         self._template_type = template_type
        
#     @property
#     def roi_template(self):
#         return self._roi_template
#     @roi_template.setter
#     def roi_template(self,roi_template):
#         self._roi_template = roi_template
        
    
#     def init_track(self,
#                    frame : np.ndarray,
#                    annot : pd.DataFrame):
        
#         # Get dimensions of the frame
#         w_frame = frame.shape[1]
#         h_frame = frame.shape[0]
        
#         # Get dimensions of the RoI
#         w_roi = annot['w'].item()
#         h_roi = annot['h'].item()
        
#         xtl_roi = annot['xtl'].item()
#         ytl_roi = annot['ytl'].item()
        
        
#         # Extract the RoI from the frame
#         roi_pix = frame[ytl_roi : ytl_roi + h_roi,
#                         xtl_roi : xtl_roi + w_roi]
        
#         # Calculate the template used for tracking from that RoI
#         self.roi_template = self._template_from_RoI(roi_pix,
#                                                     annot)
        
    
#     def track(self,
#               frame : np.ndarray):
        
#         # Check if tracker has been initialized yet
#         if self.roi_template is None:
#             raise UserWarning('Track has not been initialized yet. \
#                               Call init_track() first!')
        
#         # Get the current template that is to be found in the frame
#         tmpl = self.roi_template
        
#         # plt.figure()
#         # plt.imshow(tmpl.array)
        
#         # Get dimensions of the frame
#         w_frame = frame.shape[1]
#         h_frame = frame.shape[0]
        
#         # Get coordinates of the RoI
#         xtl_roi = tmpl.annot['xtl'].item()
#         ytl_roi = tmpl.annot['ytl'].item()

#         xbr_roi = tmpl.annot['xbr'].item()
#         ybr_roi = tmpl.annot['ybr'].item()
        
#         # Determine the detection window size from the RoI
#         w_dw = xbr_roi - xtl_roi
#         h_dw = ybr_roi - ytl_roi
        
#         # Extract the search region from the frame, which is centered around 
#         # the last known Region of Interest
#         xtl_sr = max(0,xtl_roi-self.n_pix_search)
#         ytl_sr = max(0,ytl_roi-self.n_pix_search)
#         xbr_sr = min(w_frame,xbr_roi+self.n_pix_search)
#         ybr_sr = min(h_frame,ybr_roi+self.n_pix_search)
        
#         search_region = frame[ytl_sr:ybr_sr,
#                               xtl_sr:xbr_sr]
        
        
#         # plt.figure()
#         # plt.imshow(search_region)
        
#         # Make an image pyramid from that search region
#         search_pyramid = self._make_image_pyramid(search_region,
#                                                   self.pyramid_levels,
#                                                   self.pyramid_scale)
        
#         # Determine which pyramid level is the "middle" level. We want to scale
#         # down the detection window size to that level. By scaling the detection
#         # window according to the middle level, objects becoming smaller can 
#         # be detected in the lower pyramid levels, and objects becoming bigger
#         # can be detected in the upper pyramid levels
#         middle_level = int(self.pyramid_levels / 2)
        
#         # Get scale factors from middle level        
#         s_x = search_pyramid[middle_level]['s_x']
#         s_y = search_pyramid[middle_level]['s_y']
        
#         # Calculate scaled template size
#         w_tmpl_scaled = int(np.rint(w_dw * s_x))
#         h_tmpl_scaled = int(np.rint(h_dw * s_y))
        
#         tmpl_scaled = tmpl
#         tmpl_scaled.array = cv2.resize(tmpl.array.astype(np.float32),
#                                         dst=None,
#                                         dsize=(w_tmpl_scaled,h_tmpl_scaled),
#                                         fx=None,
#                                         fy=None,
#                                         interpolation=cv2.INTER_LINEAR).astype(tmpl.array.dtype)
        
#         # plt.figure()
#         # plt.imshow(tmpl_scaled.array)
        
#         # Now search for the scaled template in all pyramid levels
#         ncc_pyramid = {}
        
#         ncc_best = -np.infty
        
#         for l in range(self.pyramid_levels):
            
#             # Calculate the NCC at all possible locations within the pyramid 
#             # level
#             ncc_map = self._ncc_map_fft(search_pyramid[l]['img'],
#                                         tmpl_scaled.array)
            
#             ncc_map_AR = self._ncc_map_fft_vAR(search_pyramid[l]['img'],
#                                                tmpl_scaled.array)
            
            
#             # Find the highest NCC value and its location (xtl,ytl) within the 
#             # pyramid level
#             idx = np.argmax(ncc_map)
#             ytl_level, xtl_level = np.unravel_index(idx, ncc_map.shape)
            
#             ncc_level = ncc_map[ytl_level,xtl_level]
            
#             # Save result, if better than previous best
#             if ncc_level > ncc_best:
                
#                 ncc_best = ncc_level
                
#                 xtl_sr_best = int(np.rint(xtl_level / search_pyramid[l]['s_x']))
#                 ytl_sr_best = int(np.rint(ytl_level / search_pyramid[l]['s_y']))
                
#                 w_sr_best = int(np.rint(w_tmpl_scaled / search_pyramid[l]['s_x']))
#                 h_sr_best = int(np.rint(h_tmpl_scaled / search_pyramid[l]['s_y']))
                
#                 roi_best = \
#                     search_pyramid[l]['img'][ytl_sr_best: ytl_sr_best + h_sr_best,
#                                              xtl_sr_best: xtl_sr_best + w_sr_best]
            
#             # Store raw result mostly for debugging purposes
#             ncc_pyramid[l] = ncc_map
            
#         # Construct a new template from the best result and return
#         xtl_best = xtl_sr + xtl_sr_best
#         ytl_best = ytl_sr + ytl_sr_best
#         xbr_best = xtl_best + w_sr_best
#         ybr_best = ytl_best + h_sr_best
        
#         annot_best = pd.DataFrame(data = [[xtl_best,
#                                            ytl_best,
#                                            xbr_best,
#                                            ybr_best,
#                                            xbr_best - xtl_best,
#                                            ybr_best - ytl_best]],
#                                   columns =  ['xtl', 'ytl', 'xbr', 'ybr','w','h'])
        
#         roi_best = RoI(roi_best,
#                        annot_best)
              
#         return roi_best,ncc_best
    
#     def _template_from_RoI(self,
#                            array : np.ndarray,
#                            annot : pd.DataFrame):
        
#         if self.template_type == 'appearance':
#             return RoI(array,annot)
#         else:
#             raise NotImplementedError()
            
#     def _make_image_pyramid(self,
#                             sr : np.ndarray,
#                             levels : int,
#                             scale : float):
        
#         # Get original dimensions of image patch
#         w_orig = sr.shape[1]
#         h_orig = sr.shape[0]
        
#         # Create dictionary to store pyramid levels in
#         pyramid = {}
        
#         # Create the image pyramid         
#         for l in range(levels):
            
#             pyramid[l] = {}
            
#             # Determine the theoretical scale factor of this level
#             s_x = scale**l
#             s_y = scale**l
            
#             # Determine the theoretical size of the image on this level 
#             w_level = w_orig * s_x
#             h_level = h_orig * s_y
            
#             # Distunguish between 0th level and all subsequent levels
#             if l == 0:
#                 # On the 0th level, the image pyramid level is just 
#                 # the original image
#                 img_pyr = sr
                                
#             else:
#                 # Subsequent levels have to be treated with more care
                
#                 # Round w_level and h_level to next integer
#                 w_level = int(np.rint(w_level))
#                 h_level = int(np.rint(h_level))
                
#                 # Recalculate the scale factors
#                 s_x = w_level / w_orig
#                 s_y = h_level / h_orig 
                
#                 # Create resized image patch
#                 img_pyr = cv2.resize(sr.astype(np.float32),
#                                      dst=None,
#                                      dsize=(w_level,h_level),
#                                      fx=None,
#                                      fy=None,
#                                      interpolation=cv2.INTER_LINEAR)
                
#                 img_pyr = np.rint(img_pyr).astype(sr.dtype)
                
#             pyramid[l]['level'] = l
#             pyramid[l]['s_x'] = s_x
#             pyramid[l]['s_y'] = s_y
#             pyramid[l]['img'] = img_pyr
            
#         return pyramid

#     def _ncc_map_fft_vAR(self,
#                          image : np.ndarray,
#                          template : np.ndarray,
#                          eps=1e-12,
#                          **kwargs) -> float:
#         """
#         Normalized cross-correlation between two same-shaped arrays.
#         If lr_flip is True, also compute NCC(a, flip_left_right(b)) and return the max.
#         """
    
#         # Convert to float arrays
#         I = np.asarray(image, dtype=np.float64)
#         T = np.asarray(template, dtype=np.float64)
    
#         # # Flatten only after flipping, so flip works on original shape
#         # def compute_single_ncc(x, y):
#         #     x = x.ravel()
#         #     y = y.ravel()
#         #     x_centered = x - x.mean()
#         #     y_centered = y - y.mean()
    
#         #     denom = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
#         #     if denom == 0:
#         #         return 0.0
    
#         #     return float(np.dot(x_centered, y_centered) / denom)

#         def compute_single_ssd(x, y):
#             x = x.ravel()
#             y = y.ravel()

#             return sum((x-y)**2)
        
#         # Get dimensions of image and template
#         w_I = I.shape[1]
#         h_I = I.shape[0]
        
#         w_T = T.shape[1]
#         h_T = T.shape[0]
        
#         # Drag T over I, compute ncc for every location
#         x_shifts = w_I - w_T + 1
#         y_shifts = h_I - h_T + 1
        
#         ncc_map = np.zeros((y_shifts,x_shifts))
        
#         for x in range(x_shifts):
#             for y in range(y_shifts):
#                 # if x == 4 and y == 4:
#                 #     plt.figure()
#                 #     plt.imshow(I[y:y+h_T, x:x+w_T])
                    
#                 #     plt.figure()
#                 #     plt.imshow(T)
                    
#                 ncc_map[y,x] = compute_single_ssd(I[y:y+h_T, x:x+w_T],
#                                                   T)
            
#         # Return the maximum score
#         return ncc_map
    
#     def _ncc_map_fft(self,
#                     image : np.ndarray,
#                     template : np.ndarray,
#                     eps=1e-12):
#         """
#         Compute normalized cross-correlation (NCC) for all valid template positions in an image.
    
#         NCC(y,x) = sum( (I_win - mean(I_win)) * (T - mean(T)) ) /
#                    ( ||I_win - mean(I_win)|| * ||T - mean(T)|| )
    
#         Parameters
#         ----------
#         image : array-like, shape (H, W)
#             Larger search image.
#         template : array-like, shape (h, w)
#             ROI / template to track.
#         eps : float
#             Stability constant to avoid division by zero.
    
#         Returns
#         -------
#         ncc : np.ndarray, shape (H-h+1, W-w+1)
#             NCC score map for each valid top-left position of the template in the image.
#         """
#         I = np.asarray(image, dtype=np.float64)
#         T = np.asarray(template, dtype=np.float64)
    
#         if I.ndim != 2 or T.ndim != 2:
#             raise ValueError("image and template must be 2D arrays")
    
#         H, W = I.shape
#         h, w = T.shape
#         if h == 0 or w == 0 or h > H or w > W:
#             raise ValueError("template must be non-empty and fit within image")
    
#         n = h * w
    
#         # Zero-mean template and its norm
#         T0 = T - T.mean()
#         T0_norm = np.linalg.norm(T0)
#         if T0_norm < eps:
#             # Constant template -> NCC is ill-defined; return zeros (or raise).
#             return np.zeros((H - h + 1, W - w + 1), dtype=np.float64)
    
#         # --- Numerator: cross-correlation sum(I_win * T0) ---
#         # Cross-correlation can be computed as convolution with flipped kernel.
#         # We want: num(y,x) = sum_{i,j} I[y+i, x+j] * T0[i,j]
#         # Equivalent to: conv2(I, flip(T0)) sampled appropriately.
#         K = np.flipud(np.fliplr(T0))
    
#         # FFT sizes (full convolution)
#         out_shape = (H + h - 1, W + w - 1)
    
#         FI = np.fft.rfftn(I, s=out_shape)
#         FK = np.fft.rfftn(K, s=out_shape)
#         conv_full = np.fft.irfftn(FI * FK, s=out_shape)
    
#         # Extract "valid" region (top-left positions where template fully fits)
#         # For full conv, valid top-left corresponds to conv_full[h-1:H, w-1:W]
#         numerator = conv_full[h - 1 : H, w - 1 : W]
    
#         # --- Denominator part 1: ||I_win - mean(I_win)|| via integral images ---
#         # Build integral images for sum and sum of squares
#         integral = np.pad(np.cumsum(np.cumsum(I, axis=0), axis=1), ((1, 0), (1, 0)), mode="constant")
#         integral2 = np.pad(np.cumsum(np.cumsum(I * I, axis=0), axis=1), ((1, 0), (1, 0)), mode="constant")
    
#         # Window sum for all positions (vectorized)
#         # sum(y,x) over I[y:y+h, x:x+w]:
#         S = integral
#         S2 = integral2
    
#         sum_win = (S[h:, w:] - S[:-h, w:] - S[h:, :-w] + S[:-h, :-w])
#         sum2_win = (S2[h:, w:] - S2[:-h, w:] - S2[h:, :-w] + S2[:-h, :-w])
    
#         mean_win = sum_win / n
#         var_win = (sum2_win / n) - mean_win * mean_win
#         var_win = np.maximum(var_win, 0.0)  # numerical safety
    
#         win_norm = np.sqrt(var_win * n)      # ||I_win - mean(I_win)||
#         denom = win_norm * T0_norm
    
#         # --- NCC ---
#         ncc = np.zeros_like(numerator)
#         valid = denom > eps
#         ncc[valid] = numerator[valid] / denom[valid]
    
#         # Optional: clip tiny numeric overshoots beyond [-1, 1]
#         ncc = np.clip(ncc, -1.0, 1.0)
        
#         return ncc
    
# class BlockTracker(Tracker):
    
#     def __init__(self,width:int,height:int,**kwargs):
        
#         """
        
#         Parameters
#         ----------
#         width : int
#             width of the frame, i.e. width of the sensor array.
#         height : int
#             height of the frame, i.e. height of the sensor array.
#         **kwargs : TYPE
#             DESCRIPTION.

#         Returns
#         -------
#         None.

#         """
        
        
#         super().__init__(width,height)
        
#         self.corner_ratio = kwargs.pop('corner_ratio',0.3)
#         self.fill_ratio = kwargs.pop('fill_ratio',0.4)
        
#         self._q_model = None
        
#         self.shifting_coords=[(-1,-1),(0,-1),(1,-1),
#                               (-1,0),(0,0),(1,0),
#                               (-1,1),(1,1),(1,1)]

        
        
#     @property
#     def q_model(self):
#         return self._q_model
        
#     @q_model.setter
#     def q_model(self,q_model):
#         self._q_model = q_model


#     def track(self,frame:np.ndarray,
#               mask:np.ndarray,
#               bbox:pd.DataFrame):
#         """
#         This function is meant to be called by other classes/functions. 
#         It applies the mean shift procedure to the provided target_region 
#         in a possibly masked frame. The results of the mean shifting are
#         then directly written to the target_region DataFrame to spare
#         external classes the bother of assignments and stuff.

#         Parameters
#         ----------
#         frame : np.ndarray
#             DESCRIPTION.
#         mask : np.ndarray
#             DESCRIPTION.
#         target_region : pd.DataFrame
#             DESCRIPTION.

#         Returns
#         -------
#         None.
        
        
#         """
        
#         # Compare the corner blocks in q_model to the neighbourhood of blocks
#         # in the provided current frame
#         shifted_blocks = {}
#         shifted_dists = {}
        
#         for corner in self.q_model['corners'].keys():
#             # Coordinates of block
#             xy = self.q_model['corners'][corner]['xy']
            
#             # Content of block
#             B = self.q_model['corners'][corner]['B']
#             shifted_block,dist = self._block_matching(frame,
#                                                      xy,
#                                                      B)
            
#             shifted_blocks[corner] = shifted_block
#             shifted_dists[corner] = dist
        
#         # Calculate new absolute coordinates from shifting coordinates
#         for corner in self.q_model['corners'].keys():
#             coords_new = self.q_model['corners'][corner]['xy']
            
#             coords_new = \
#                 (coords_new[0]+shifted_blocks[corner][0],
#                  coords_new[1]+shifted_blocks[corner][1])

#             self.q_model['corners'][corner]['xy'] = coords_new
        
#         # Given these coordinates and the block sizes, the updated bounding box
#         # can be calculated
#         blocks = self.q_model['corners']
        
#         block_w = self.q_model['b_size'][0]
#         block_h = self.q_model['b_size'][1]
        
#         xtl,ytl = blocks['tl']['xy'][0],blocks['tl']['xy'][1]
#         xbl,ybl = blocks['bl']['xy'][0],blocks['bl']['xy'][1]
#         xbr,ybr = blocks['br']['xy'][0],blocks['br']['xy'][1]
#         xtr,ytr = blocks['tr']['xy'][0],blocks['tr']['xy'][1]
        
       
        
#         # Assign new coordinates to bboxes and return
#         bbox['xtl'] = min(xtl,xbl)
#         bbox['xbr'] = max(xbr,xtr) + block_w
        
#         bbox['ytl'] = min(ytl,ytr)
#         bbox['ybr'] = max(ybr,ybl) + block_h
        
#         bbox['w'] = bbox['xbr'] - bbox['xtl']
#         bbox['h'] = bbox['ybr'] - bbox['ytl']
        
#         return bbox
        
        
        
#     def estimate_target(self,frame:np.ndarray,
#                         mask:np.ndarray,
#                         target_region:pd.DataFrame):
        
        
#         """
#         Estimates the target model, i.e. a model of the provided region of 
#         interest that is supposed to be tracked.
        
#         The model is not returned, but written to the attribute self.q_model

#         Parameters
#         ----------
#         frame : np.ndarray
#             width x height array of pixel intensities.
#         mask: np.ndarray
#             mask for frame, that masks out all pixels below a threshold
#         bbox : pd.DataFrame
#             The annotation containing coordinates of the bounding box that
#             frame the object to track

#         Returns
#         -------
#         None
#         """
        
        
#         # blablabla
#         roi,roi_mask = self.get_ROI(frame,
#                                     mask,
#                                     target_region)
        
#         # Get corner blocks of the roi
#         tl = (target_region['xtl'].item(),target_region['ytl'].item())
        
#         corners = self._get_corner_blocks(roi.copy(),roi_mask,tl)
        
#         # If corners are None, i.e. any corner could not be found, set q_model 
#         # to non to indicate process failed
#         if corners is None:
#             self.q_model = None
            
#         else:
#             # The corners are the model, these will be tracked through the 
#             # sequence
#             self.q_model = {}
#             self.q_model['corners'] = corners
            
#             # Store block size for easy reference (w,h)
#             block_size = corners['tl']['B'].shape
#             self.q_model['b_size'] = (block_size[1],block_size[0])
        
#         return None
        
        
#     def _block_matching(self,frame:np.ndarray,tl:tuple,block:np.ndarray)->tuple:
#         """
#         Looks for best matching block to a given block in a frame and returns
#         the upper left corner coordinates (xtl,ytl) of the best match

#         Parameters
#         ----------
#         frame : np.ndarray
#             DESCRIPTION.
#         tl : tuple
#             DESCRIPTION.
#         block : np.ndarray
#             DESCRIPTION.

#         Returns
#         -------
#         tl_match : TYPE
#             DESCRIPTION.

#         """
        
        
#         # Get size of block (w,h)
#         block_size = (block.shape[1],block.shape[0])
        
#         # Get size of frame
#         frame_size = (frame.shape[1],frame.shape[0])
        
#         # Dict for storing distances of candiate blocks to reference block
#         dist = {}

#         # Compare every block in frame in neighborhood of 1 pixel to the 
#         # given block
#         for shift in self.shifting_coords:
            
#             xtl,ytl = tl[0] + shift[0] , tl[1] + shift[1]
#             xbr = tl[0] + block_size[0] + shift[0]
#             ybr = tl[1] + block_size[1] + shift[1]
            
#             # Check if shifted block is inside frame
#             if all(c >= 0 for c in [xtl,ytl]) \
#                 and xbr <= frame_size[0] \
#                     and ybr <= frame_size[1]:
                
#                 # Get candidate block from frame
#                 cand_block = frame[ytl:ybr,xtl:xbr]
                
#                 # Calculate distance to reference block
#                 dist[shift] = self._calc_MAD(block,cand_block)
#                 # dist[(xtl,ytl)] = self._calc_MAD(block,cand_block)

        
#         # Find the minimal distance and return the corresponding shift 
#         # of the matching block w.r.t. the reference block
#         shift_match = min(dist, key=dist.get)
        
#         return shift_match, dist       
        
                
#     def _calc_MAD(self,ref_block:np.ndarray,cand_block:np.ndarray):
#         """
#         Calculates the not-normalized absolute difference between two blocks        

#         Parameters
#         ----------
#         ref_block : np.ndarray
#             DESCRIPTION.
#         cand_block : np.ndarray
#             DESCRIPTION.

#         Returns
#         -------
#         None.

#         """            
        
#         MAD = abs(ref_block - cand_block).sum()
        
#         return MAD
        
#     def _get_corner_blocks(self,roi:np.ndarray, roi_mask:np.ndarray,
#                            tl_frame:tuple):
#         """
        

#         Parameters
#         ----------
#         frame : np.ndarray
#             width x height array of pixel intensities.
#         roi_mask : np.ndarray
#             DESCRIPTION.
#         tl_frame: tuple
#             top left corner coordinates (xtl,ytl) of the region of interest 
#             roi in the frame

#         Returns
#         -------
#         None.

#         """
        
#         # coordinates of roi in frame
#         xtl = tl_frame[0]
#         ytl = tl_frame[1]
        
#         # required percentage of unmasked values in corner block
#         fill_factor = self.fill_ratio
        
#         # Size of roi
#         roi_h,roi_w = roi_mask.shape
        
#         # Block size (w,h) of corner blocks is 30 % of dimensions
#         block_w = max(2,int(self.corner_ratio*roi_mask.shape[1]))
#         block_h = max(2,int(self.corner_ratio*roi_mask.shape[0]))
#         block_size = (block_w,block_h)
        
#         # dictionary for storing coordinates of corner blocks
#         corners = {'tl':{},'bl':{},'br':{},'tr':{}}
        
#         ##################### Top left ########################################
#         tl_roi = self._find_corner(roi_mask,
#                                block_size,
#                                (0,0),
#                                (1,1),
#                                fill_factor)
        
#         ##################### Bottom left #####################################
#         bl_roi = self._find_corner(roi_mask,
#                                block_size,
#                                (0,roi_h-block_h),
#                                (1,-1),
#                                fill_factor)
               
#         ##################### Bottom right ####################################
#         br_roi = self._find_corner(roi_mask,
#                                block_size,
#                                (roi_w-block_w,roi_h-block_h),
#                                (-1,-1),
#                                fill_factor)
               
#         ##################### Top right #######################################
#         tr_roi = self._find_corner(roi_mask,
#                                block_size,
#                                (roi_w-block_w,0),
#                                (-1,1),
#                                fill_factor)
        
#         # Check if any of the corners is None, i.e. could not be found
#         if any([c is None for c in [tl_roi,bl_roi,br_roi,tr_roi]]):
#             return None
        
#         # If all corners were found, write them to dictionary in the following
#         # order: [top left, bottom left, bottom_right, top right]
        
#         corners['tl']['xy'] = (tl_roi[0] + xtl, tl_roi[1] + ytl)
        
#         corners['tl']['B'] = roi[tl_roi[1]:tl_roi[1]+block_h,
#                                 tl_roi[0]:tl_roi[0]+block_w]

#         corners['bl']['xy'] = (bl_roi[0] + xtl, bl_roi[1] + ytl)
        
#         corners['bl']['B'] = roi[bl_roi[1]:bl_roi[1]+block_h,
#                                 bl_roi[0]:bl_roi[0]+block_w]
        
#         corners['br']['xy'] = (br_roi[0] + xtl, br_roi[1] + ytl)
        
#         corners['br']['B'] = roi[br_roi[1]:br_roi[1]+block_h,
#                                 br_roi[0]:br_roi[0]+block_w]

#         corners['tr']['xy'] = (tr_roi[0] + xtl, tr_roi[1] + ytl)
        
#         corners['tr']['B'] = roi[tr_roi[1]:tr_roi[1]+block_h,
#                                 tr_roi[0]:tr_roi[0]+block_w]
        
        
#         return corners
        
            
            
#     def _find_corner(self,roi_mask:np.ndarray,
#                      size:tuple,
#                      start:tuple,
#                      step:tuple,
#                      fill_thresh:float)->tuple:
#         """
        

#         Parameters
#         ----------
#         roi_mask : np.ndarray
#             DESCRIPTION.
#         size : tuple
#             required size of corner blocks
#         start : tuple
#             (xtl,ytl) coordinates of where to start looking for a corner.
#         step : tuple
#             (dx,dy) step to step in when looking for a corner.
#         fill_thresh : float
#             Percentage of unmasked pixels required to be in corner block.

#         Returns
#         -------
#         tuple
#             (xtl,ytl) coordinates of found corner block.

#         """

#         # dimensions of roi
#         roi_h, roi_w = roi_mask.shape
        
#         # number of pixels in block
#         N_pix = size[0] * size[1]
        
#         # Start search pocedure
#         filled = False
        
#         # Determine a maximum number of iterations, after which to 
        
#         xtl = start[0]
#         ytl = start[1]
        
#         while filled == False:
            
#             # Calculate number of non-zero pixels in mask
#             fill_q = roi_mask[ytl:ytl+size[1],xtl:xtl+size[0]].sum() / N_pix
            
#             # If corner block is sufficiently filled, stop procedure
#             if fill_q>=fill_thresh:
#                 filled = True
#                 break
            
#             # Else move the block
#             else:
#                 # If corner block is still in the correct half, increment xtl
                
#                 # Differentiate between looking for right and left corners
#                 if step[0] > 0 and xtl + size[0] < roi_w*0.75:
#                     xtl = xtl + step[0]
#                 elif step[0] < 0 and xtl > roi_w*0.25 :
#                     xtl = xtl + step[0]
#                 else:
#                     xtl = start[0]
#                     ytl = ytl + step[1]
                    
#             # Break condition to avoid endless loop if finding a corner is 
#             # not possible
#             if step[1] > 0 and ytl > roi_h/2:
#                 return None
#             if step[1] < 0 and ytl < roi_h/2:
#                 return None
            
#         return (xtl,ytl)
        
# class BlockTracker_quarter(BlockTracker):

#     def _get_corner_blocks(self,roi:np.ndarray, roi_mask:np.ndarray,
#                            tl_frame:tuple):
#         """
        

#         Parameters
#         ----------
#         frame : np.ndarray
#             width x height array of pixel intensities.
#         roi_mask : np.ndarray
#             DESCRIPTION.
#         tl_frame: tuple
#             top left corner coordinates (xtl,ytl) of the region of interest 
#             roi in the frame

#         Returns
#         -------
#         None.

#         """
        
#         # coordinates of roi in frame
#         xtl = tl_frame[0]
#         ytl = tl_frame[1]
        
#         # required percentage of unmasked values in corner block
        
#         # Size of roi
#         roi_h,roi_w = roi_mask.shape
        
#         # Size of blocks
#         block_w = int(roi_w/2)
#         block_h = int(roi_h/2)
        
#         # Coordinates of upper left corner of blocks
#         tl_roi = (0,0)
#         bl_roi = (0,roi_h-block_h)
#         br_roi = (roi_w-block_w,roi_h-block_h)
#         tr_roi = (roi_w-block_w,0)
               
#         # dictionary for storing coordinates of corner blocks
#         corners = {'tl':{},'bl':{},'br':{},'tr':{}}

        
#         # If all corners were found, write them to dictionary in the following
#         # order: [top left, bottom left, bottom_right, top right]
        
#         corners['tl']['xy'] = (tl_roi[0] + xtl, tl_roi[1] + ytl)
        
#         corners['tl']['B'] = roi[tl_roi[1]:tl_roi[1]+block_h,
#                                 tl_roi[0]:tl_roi[0]+block_w]

#         corners['bl']['xy'] = (bl_roi[0] + xtl, bl_roi[1] + ytl)
        
#         corners['bl']['B'] = roi[bl_roi[1]:bl_roi[1]+block_h,
#                                 bl_roi[0]:bl_roi[0]+block_w]
        
#         corners['br']['xy'] = (br_roi[0] + xtl, br_roi[1] + ytl)
        
#         corners['br']['B'] = roi[br_roi[1]:br_roi[1]+block_h,
#                                 br_roi[0]:br_roi[0]+block_w]

#         corners['tr']['xy'] = (tr_roi[0] + xtl, tr_roi[1] + ytl)
        
#         corners['tr']['B'] = roi[tr_roi[1]:tr_roi[1]+block_h,
#                                 tr_roi[0]:tr_roi[0]+block_w]
        
        
#         return corners    

# class meanShift_base():

#     def __init__(self,width:int,height:int,**kwargs):
        
#         """
        
    
#         Parameters
#         ----------
#         width : int
#             width of the frame, i.e. width of the sensor array.
#         height : int
#             height of the frame, i.e. height of the sensor array.
#         **kwargs : TYPE
#             DESCRIPTION.
    
#         Returns
#         -------
#         None.
    
#         """
        
        
#         self._width = int(width)    
#         self._height = int(height)
        
#         self._q_model = None
        
#         self._histSize = kwargs.pop('histSize',30)
        
#     @property
#     def width(self):
#         return self._width
    
#     @property
#     def height(self):
#         return self._height
    
#     @property
#     def histSize(self):
#         return self._histSize
    
#     @property
#     def q_model(self):
#         return self._q_model
    
#     @q_model.setter
#     def q_model(self,q_model):
#         self._q_model = q_model
    
    
#     def get_ROI(self,frame:np.ndarray,
#                 mask:np.ndarray,
#                 region:pd.DataFrame):
        
#         # extract the pixels of a region of interest from a frame whose 
#         # coordinates are given in region. Also extract the pixels of the
#         # mask for that RoI
        
#         # Get the bounding box coordinates
#         xtl = region['xtl'].item(); ytl = region['ytl'].item()
#         xbr = region['xbr'].item(); ybr = region['ybr'].item()
                
#         # Mask the frame and get all non-masked pixels inside the bounding box
#         roi_mask = mask[ytl:ybr,xtl:xbr].astype(np.uint8)
#         roi = frame[ytl:ybr,xtl:xbr]
        
#         return roi, roi_mask

# class meanShift_CV(meanShift_base):

#     def __init__(self,width:int,height:int,**kwargs):
#         """
        

#         Parameters
#         ----------
#         width : int
#             width of the frame, i.e. width of the sensor array.
#         height : int
#             height of the frame, i.e. height of the sensor array.
#         **kwargs : TYPE
#             DESCRIPTION.

#         Returns
#         -------
#         None.

#         """
        
#         super().__init__(width,height,**kwargs)
        
#     def estimate_target(self,frame:np.ndarray,
#                         mask:np.ndarray,
#                         target_region:pd.DataFrame):
#         """
#         Estimates the target model, i.e. a model of the provided region of 
#         interest that is supposed to be tracked.
        
#         The model is not returned, but written to the attribute self.q_model

#         Parameters
#         ----------
#         frame : np.ndarray
#             width x height array of pixel intensities.
#         bbox : pd.DataFrame
#             The annotation containing coordinates of the bounding box that
#             frame the object to track

#         Returns
#         -------
#         None

#         """

#         # Get range of pixel intensities of the whole image
#         hist_range = [(np.min(frame)).astype(np.uint16),
#                       (np.max(frame)+1).astype(np.uint16)]
                
#         # Get the region of interest and a mask for the ROI
#         roi,roi_mask = self.get_ROI(frame,
#                                     mask,
#                                     target_region)
        
        
#         # Calculate the Histogram for the RoI
#         roi_hist = cv2.calcHist([roi],
#                                 [0],
#                                 roi_mask,
#                                 [self.histSize],
#                                 hist_range,
#                                 False)

#         # The histogram is the target_model, i.e. the model of the target 
#         # region that the tracker is supposed to follow
#         self.q_model = {'roi_hist':roi_hist,
#                         'hist_range':hist_range}
        
        
#     def track(self,frame:np.ndarray,
#               mask:np.ndarray,
#               bbox:pd.DataFrame):
        
        
        
#         # Convert bounding box into OpenCV Rectangle (xtl,ytl,w,h)
#         xtl,ytl = bbox['xtl'].item(), bbox['ytl'].item()
#         xbr,ybr = bbox['xbr'].item(), bbox['ybr'].item()
        
#         tracking_window = (xtl,ytl,xbr-xtl,ybr-ytl)
        
#         # Mask the frame: Search only within a 1 Pixel range for the shifted
#         # bounding box
#         mask = np.zeros(frame.shape)
#         mask[ytl-1:ybr+2,xtl-1:xbr+2] = 1
        
#         frame = (frame*mask).astype(np.uint16)
        
#         # Calculate the backprojection of the target model (q_model) on
#         # the frame
#         B = cv2.calcBackProject([frame],
#                                 [0],
#                                 self.q_model['roi_hist'],
#                                 self.q_model['hist_range'],
#                                 1)
        
#         # Apply mean shift tracking only for a single iteration
#         term_crit = (cv2.TERM_CRITERIA_COUNT, 1,0)
        
#         # Apply mean shift on the backprojected image
#         updated_window = cv2.meanShift(B,
#                                        tracking_window,
#                                        term_crit)
        
#         print('Implement tracking function for opencv variant')
        
#         # hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
#         # dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)
#         # # apply meanshift to get the new location
#         # ret, track_window = cv.meanShift(dst, track_window, term_crit)
#         # # Draw it on image
#         # x,y,w,h = track_window
#         # img2 = cv.rectangle(frame, (x,y), (x+w,y+h), 255,2)



    
    
    

# class KhanBackprojection_CV(meanShift_base):
    
    

#     """
#     Based on the publication 
    
#     A back projection scheme for accurate mean shift based tracking
#     Conference Paper in Proceedings / ICIP 
#     International Conference on Image Processing · September 2010
#     DOI: 10.1109/ICIP.2010.5652577 · Source: DBLP
#     """    

#     def __init__(self,width:int,height:int,**kwargs):
#         """
        

#         Parameters
#         ----------
#         width : int
#             width of the frame, i.e. width of the sensor array.
#         height : int
#             height of the frame, i.e. height of the sensor array.
#         **kwargs : TYPE
#             DESCRIPTION.

#         Returns
#         -------
#         None.

#         """
        
#         self.shifting_coords=[(-1,-1),(0,-1),(1,-1),
#                               (-1,0),(0,0),(1,0),
#                               (1,-1),(1,0),(1,1)]

        
#         super().__init__(width,height,**kwargs)
        
#     def estimate_target(self,frame:np.ndarray,
#                         mask:np.ndarray,
#                         target_region:pd.DataFrame):
#         """
#         Estimates the target model, i.e. a model of the provided region of 
#         interest that is supposed to be tracked.
        
#         The model is not returned, but written to the attribute self.q_model

#         Parameters
#         ----------
#         frame : np.ndarray
#             width x height array of pixel intensities.
#         bbox : pd.DataFrame
#             The annotation containing coordinates of the bounding box that
#             frame the object to track

#         Returns
#         -------
#         None

#         """

#         # Get range of pixel intensities of the whole image
#         hist_range = [(np.min(frame)).astype(np.uint16),
#                       (np.max(frame)+1).astype(np.uint16)]
                
#         # Get the region of interest and a mask for the ROI
#         roi,roi_mask = self.get_ROI(frame,
#                                     mask,
#                                     target_region)
        
        
#         # Calculate the Histogram for the RoI
#         roi_hist = cv2.calcHist([roi],
#                                 [0],
#                                 roi_mask,
#                                 [self.histSize],
#                                 hist_range,
#                                 False)

#         # The histogram is the target_model, i.e. the model of the target 
#         # region that the tracker is supposed to follow
#         self.q_model = {'roi_hist':roi_hist,
#                         'hist_range':hist_range}
        
        
#     def track(self,frame:np.ndarray,
#               mask:np.ndarray,
#               bbox:pd.DataFrame):
        
#         # Get the current window
#         xtl,ytl = bbox['xtl'].item(), bbox['ytl'].item()
#         xbr,ybr = bbox['xbr'].item(), bbox['ybr'].item()
        
#         tl = np.array([xtl,ytl])
#         br = np.array([xbr,ybr])
        
#         w = xbr-xtl
#         h = ybr-ytl
        
#         tracking_window = [xtl,ytl,xbr-xtl,ybr-ytl]
        
#         # Produce a number of candidate windows from the current 
#         # tracking_window by shifting it in all possible directions
#         candidate_windows = []
#         ROIS = []
#         roi_hists = []
#         sim = []
#         dbug = []
        
#         for shift in self.shifting_coords:
            
#             bbox_shift = bbox.copy()
#             bbox_shift['xtl'] = bbox_shift['xtl'] + shift[0]
#             bbox_shift['ytl'] = bbox_shift['ytl'] + shift[1]
#             bbox_shift['xbr'] = bbox_shift['xbr'] + shift[0]
#             bbox_shift['ybr'] = bbox_shift['ybr'] + shift[1]
            
#             roi,roi_mask = self.get_ROI(frame,
#                                         mask,
#                                         bbox_shift)
            
#             candidate_windows.append(bbox_shift)
            
#             ROIS.append(roi)
            
#             hist = cv2.calcHist([roi],
#                                 [0],
#                                 roi_mask,
#                                 [self.histSize],
#                                 self.q_model['hist_range'],
#                                 False)
            
#             # hist = np.divide(self.q_model['roi_hist'],hist,
#             #                             out=np.ones_like(hist),
#             #                             where=hist!=0)
            
#             roi_hists.append(hist)
            
#             # print(np.linalg.norm(hist-self.q_model['roi_hist']/(w*h)))
#             # print(self._Bhatta_dist(hist,self.q_model['roi_hist']/(w*h)))
#             # print('\n')
#             b_coeff = self._bhatta_coeff(self.q_model['roi_hist'],
#                                               hist)
#             dbug.append(b_coeff)
#             sim.append(b_coeff-0.02*np.linalg.norm(shift))
        
#         # Concatenate candidate windows to dataframe
#         # candidate_windows = pd.concat(candidate_windows)
#         # candidate_windows = candidate_windows.reset_index(drop=False)
#         # Find maximal Bhattacharya coefficient and return corresponding 
#         # shifted region
#         idx_max = np.argmax(sim)
#         # bbox_shift = candidate_windows.loc[[idx_max]]
#         bbox[bbox.columns] = candidate_windows[idx_max][bbox.columns]
        
#         return bbox


#     def _hellinger_dist(self,hist1,hist2):
        
#         hist1 = np.array(hist1).flatten()
#         hist2 = np.array(hist2).flatten()

#         return 1-sum(np.sqrt(hist1*hist2))
    
#     def _bhatta_coeff(self,hist1,hist2):
        
#         # Convert to equal shape
#         hist1 = np.array(hist1).flatten()
#         hist2 = np.array(hist2).flatten()
        
#         # Normalize histograms
#         hist1 = hist1/sum(hist1)
#         hist2 = hist2/sum(hist2)
               
#         return sum(np.sqrt(hist1*hist2))
        
# class meanShift_ar_adaptive(meanShift_base):
    
    
#     def __init__(self,width:int,height:int,**kwargs):
#         """
        

#         Parameters
#         ----------
#         width : int
#             width of the frame, i.e. width of the sensor array.
#         height : int
#             height of the frame, i.e. height of the sensor array.
#         b_width : int
#             bandwidth of the kernel to use right after initialization. usually
#             this is 1 since the bandwidth of the target model is normalized to
#             1.
#         q_model: dict
#             target model
#         histSize: int
#             number of bins to use in the histogram of target model and 
#             candidate model.
#         Nmax : int
#             maximal number of iterations to apply mean shift for.
#         eps: float
#             if the mean is not shifted by at least eps pixels, then the mean
#             shift procedure is stopped.
#         minsize: int
#             mean shift tracking tends to produce smaller and smaller target
#             regions to achieve higher similarity between target and candidate
#             model. To prevent this from happening, a minimal size can be
#             provided. Neither the width nor height of the candidate region can
#             fall below that limit.
#         r: float
#             To avoid the scale changing to rapidly (i.e. from frame to frame) 
#             the bandwidth is filtered via the rule 
#             b_width = r * b_new + (1-r) * b_old
#             That means higher r allow for more rapid changes in scale, while 
#             lower r hamper rapid changes. 
#         Returns
#         -------
#         None.

#         """
        
#         super().__init__(width,height,**kwargs)
        
#         self._b_width = kwargs.pop('b_width',1)     
#         self._Nmax = kwargs.pop('Nmax',10)
#         self._eps = kwargs.pop('eps',1)
#         self._minsize = kwargs.pop('minsize',4)
#         self._r = kwargs.pop('r',0.05)
        
#         # super().__init__(width,height,**kwargs)

    
#     # @property
#     # def width(self):
#     #     return self._width
    
#     # @property
#     # def height(self):
#     #     return self._height
    
#     # @property
#     # def histSize(self):
#     #     return self._histSize
    
#     # @property
#     # def q_model(self):
#     #     return self._q_model
    
#     # @q_model.setter
#     # def q_model(self,q_model):
#     #     self._q_model = q_model

#     @property
#     def b_width(self):
#         return self._b_width
    
#     @b_width.setter
#     def b_width(self,b_width):
#         self._b_width = self.r * b_width + (1-self.r) * self.b_width
        
#     @property
#     def histSize(self):
#         return self._histSize
    
#     @property
#     def Nmax(self):
#         return self._Nmax
    
#     @property
#     def eps(self):
#         return self._eps

#     @property
#     def minsize(self):
#         return self._minsize 

#     @property
#     def r(self):
#         return self._r  
    
    
#     def track(self,frame:np.ndarray,
#                    mask:np.ndarray,
#                    target_region:pd.DataFrame):
#         """
#         This function is meant to be called by other classes/functions. 
#         It applies the mean shift procedure to the provided target_region 
#         in a possibly masked frame. The results of the mean shifting are
#         then directly written to the target_region DataFrame to spare
#         external classes the bother of assignments and stuff.

#         Parameters
#         ----------
#         frame : np.ndarray
#             DESCRIPTION.
#         mask : np.ndarray
#             DESCRIPTION.
#         target_region : pd.DataFrame
#             DESCRIPTION.

#         Returns
#         -------
#         None.
        
        
#         """
        
#         shift_res = self.meanShift(frame,
#                                    mask,
#                                    target_region.copy())
        

#         if shift_res is None:
#             # if the shift failed, shift_res will be None, return the original
#             # target region in that case
#             pass
        
#         else:
#             # Replace the initial target region with the shifted/scaled region
#             target_region[target_region.columns] = \
#                 shift_res['target_region'][target_region.columns]
                
#             # Add the last (i.e. optimal) value of the Bhattacharyya coefficient. 
#             # Describes the similarity between the target and candidate model
#             target_region['sim_meas'] = shift_res['p_coeff'][-1]
        
#         return target_region

        
#     def estimate_target(self,frame:np.ndarray,
#                              mask:np.ndarray,
#                              bbox:pd.DataFrame):
#         """
#         Estimates the target model, i.e. the model of the pixels to track.
        
#         The model is not returned, but written to the attribute self.q_model

#         Parameters
#         ----------
#         frame : np.ndarray
#             width x height array of pixel intensities.
#         mask : np.ndarray
#             width x height boolean array.
#         bbox : pd.DataFrame
#             The annotation containing coordinates of the bounding box that
#             frame the object to track

#         Returns
#         -------
#         None

#         """
        
#         # Get range of pixel intensities of the whole image
#         ranges = [np.min(frame),np.max(frame)]

#         # Bandwidth of target model is 1
#         b_width = 1
        
#         # Get height and width of the bounding box
#         w = bbox['xbr'].item() - bbox['xtl'].item()
#         h = bbox['ybr'].item() - bbox['ytl'].item() 
        
#         # Determine normalization constants hy and hx
#         hy = h/2
#         hx = w/2 
            
#         # Estimate the target model
#         q_model = self.estimate_model(frame,mask,bbox,
#                                       ranges = ranges,
#                                       norm_const = (hx,hy),
#                                       b_width = b_width )
        
#         # Save target model as attribute
#         self.q_model = q_model
        
#         return None
    
#     def estimate_candidate(self,frame:np.ndarray,
#                            mask:np.ndarray,
#                            bbox:pd.DataFrame)->dict:
#         """
#         Estimates the candidate model, i.e. the model of the pixels to compare
#         to the target model.
        
        

#         Parameters
#         ----------
#         frame : np.ndarray
#             width x height array of pixel intensities.
#         mask : np.ndarray
#             width x height boolean array.
#         bbox : pd.DataFrame
#             The annotation containing coordinates of the bounding box that
#             frames the candidate region in which the object is assumed to be.

#         Returns
#         -------
#         dict
#             The candidate model.
        

#         """
        
#         # Normalization of the candiate is always done w.r.t. the target model
#         ranges = self.q_model['ranges']
#         (hx,hy) = self.q_model['norm_const']
        
        
#         p_model = self.estimate_model(frame,mask,bbox,ranges,(hx,hy),
#                                       bbox['b_width'].item())
        
#         return p_model
        
#     def estimate_model(self,frame:np.ndarray,mask:np.ndarray,bbox,
#                        ranges,norm_const,b_width)->dict:
#         """
        

#         Parameters
#         ----------
#         frame : np.ndarray
#             width x height array of pixel intensities.
#         mask : np.ndarray
#             width x height boolean array.
#         bbox : pd.DataFrame
#             The annotation containing coordinates of the target region
#             or candidate region.
#         ranges : list
#             List containing the minimal and maximal pixel intensities.
#         norm_const : TYPE
#             DESCRIPTION.
#         b_width : TYPE
#             DESCRIPTION.

#         Returns
#         -------
#         dict
#             DESCRIPTION.

#         """

#         # Unpack tuple
#         (hx,hy) = norm_const
           
#         # Get the bounding box coordinates
#         xtl = bbox['xtl'].item()
#         ytl = bbox['ytl'].item()
#         xbr = bbox['xbr'].item()
#         ybr = bbox['ybr'].item()
        
#         # Get height and width of the bounding box
#         w = xbr - xtl
#         h = ybr - ytl 
        
#         # Mask the frame and get all non-masked pixels inside the bounding box
#         mask = mask[ytl:ybr,xtl:xbr]
#         region = frame[ytl:ybr,xtl:xbr]
        
#         # Get coordinates of all pixels in bounding box
#         y_coord, x_coord = np.indices(region.shape)
        
#         # region = region[mask]
#         x_coord = x_coord[mask]
#         y_coord = y_coord[mask]
        
#         # Normalize all pixel coordinates by centering and scaling to 
#         # "radius" 1
#         x_coord = (x_coord - w/2) / hx
#         y_coord = (y_coord - h/2) / hy
        
#         pix_coords = np.vstack([x_coord,y_coord]).T
        
#         # Now calculate a histogram of the regions pixel intensities
#         H = cv2.calcHist(images=[np.float32(region)],
#                          channels=[0],
#                          mask=np.uint8(mask),
#                          histSize=[self.histSize],
#                          ranges = np.float32(ranges))
        
#         # Map every pixel in the region to its appropriate bin
#         # Calculate bin width
#         bin_width = ((ranges[1] - ranges[0]) / self.histSize)
#         # Normalize pixel values to 0,...,1 
#         bin_map = (region[mask] - ranges[0])
#         # Normalized pixel module bin width yiels the index of the  
#         # corresponding bin
#         bin_map = (bin_map) // bin_width
        
#         # The largest intensity value may spill over to a nonexistent bin
#         # Threshold the maximal bin value  
#         bin_map = [int(min(b,self.histSize-1)) for b in bin_map]
        
#         # The lowest value may be just below ranges[0] and spill into a
#         # nonexistent -1-bin 
#         bin_map = [int(max(b,0)) for b in bin_map]
#         # Calculate the model, i.e. the probability of each pixel
#         # of being in the model
        
#         # Initialize an empty model, one entry in dictionary for every bin
#         model = {p:[] for p in range(self.histSize)}
        
#         # Initialize a variable for the normalization constant
#         C = 0

#         # Loop over all pixels
#         for p in range(len(region[mask])):
            
#             # Get normalized coordinates of that pixel
#             x = pix_coords[p,:]
            
#             # Get corresponding bin
#             hist_bin = bin_map[p]
              
#             # Calculate k(x^2)
#             kE = self._epanechnikov_kernel(x,b_width)
#             model[hist_bin].append(kE)  
#             C = C + kE
        
#         # It may happen that C is zero, e.g. if no unmasked pixels were in the
#         # target region, or all unmasked pixels where weighted with zero by 
#         # the kernel (if too far from the center)
#         # in these cases return None to indicate, that building a model failed
#         if C == 0:
#             return None
        
#         # Normalize the model
#         for q in model.keys():
#             model[q] = sum(model[q])/C

#         # Return normalization constants, bandwidth, etc
#         return {'model':model,'ranges':ranges,'norm_const':(hx,hy),
#                 'b_width':b_width,'bin_map':bin_map,'x':pix_coords,
#                 'hist':H}
    
#     def meanShift(self,frame:np.ndarray,
#                   mask:np.ndarray,
#                   target_region:pd.DataFrame)->dict:
#         """
#         Applies the mean shift procedure on three different scales to the 
#         provided target region.

#         frame : np.ndarray
#             width x height array of pixel intensities.
#         mask : np.ndarray
#             width x height boolean array.
#         target_region : pd.DataFrame
#             The annotation containing coordinates of the target region.

#         Returns
#         -------
#         res_opt : dict
#             Dictionary containing the results of the mean shift procedure
#             with the following keys:
                
#             'p_coeff': List of Bhattacharyya coefficients over all iterations
#             'target_region': The shifted/scaled target region
#             'meanShift': The mean shift vector
#             'p_model': The candidate model of the shifted/scaled target region

#         """
        
#         # Remember the original index to reinstate it when returning results
#         orig_idx = target_region.index.name
        
#         # Take the provided target region, and create three versions with three
#         # different associated bandwidths (smaller, same, larger)
#         proposed_regions = self._propose_target_regions(target_region.copy())
        
#         # Calculate bandwidth of the target region
#         target_region['bwidth'] = self._calc_bwidth(target_region)
        
#         # Initialize a dictionary to store results for all bandwiths in
#         res = {}
        
#         # list for storing best Bhattacharyya coefficient per proposed region in
#         p_coeff_opt = []
        
#         for idx in proposed_regions.index:
                        
#             # Get the target region
#             target = proposed_regions.loc[[idx]]
            
#             # Calculate the initial model for the target region
#             p = self.estimate_candidate(frame,
#                                         mask,
#                                         target)
            
#             # If no model could be estimated, skip this iteration
#             if p is None:
#                 continue
            
#             # In all other cases apply meanshift
            
#             # inizialize dictionary for results of this iteration
#             res[idx] = {}
        
#             # Initialize a list for storing the Bhattacharyya coefficients in
#             p_coeff = []
            
#             # Calculate Bhattacharyya coefficient for first iteration
#             p_coeff.append(self._bhattacharyya_coefficient(self.q_model,
#                                                            p))
            
#             # Initialize a variable for summing up the mean shift vectors
#             y_sum = np.zeros(2)
            
#             for i in range(self.Nmax):
                                    
#                 # Calculate weights of pixels in target region
#                 w = self._calc_weights(p,self.q_model)
            
#                 # Shift mean
#                 y = self._calc_weighted_mean(p,w)
            

#                 # Update target region by applying the shift vector
#                 _,target_temp = self._update_target_region(target.copy(),
#                                                            y=y)
                
#                 # Calculate a new candidate model for the shifted region
#                 p_temp = self.estimate_candidate(frame,
#                                                  mask,
#                                                  target_temp)
                
#                 # Calculate Bhattacharyya of shifted region's model
#                 p_coeff_temp = self._bhattacharyya_coefficient(self.q_model,
#                                                                p_temp)  
                
#                 # if the similarity did not increase after the shift, end
#                 # optimization
#                 if not (p_coeff_temp > p_coeff[-1]):
#                     break
                
#                 p_coeff.append(p_coeff_temp)
#                 p = p_temp
#                 target = target_temp
#                 y_sum = y_sum + y
                
#                 # if shift of mean is less than threshold, end optimization
#                 if np.linalg.norm(y) < self.eps:
#                     break

#             # Store results of this iteration
#             res[idx]['p_coeff'] = p_coeff
#             res[idx]['target_region'] = target
#             res[idx]['meanShift'] = y_sum
#             res[idx]['p_model'] = p      
        
#             # Append last Bhattacharyya coefficient to a list too look for
#             # optimum in
#             p_coeff_opt.append(p_coeff[-1])
        
#         # Check if result dictionary is non-empty
#         if len(res) != 0:
        
#             # Get best result (highest Bhattacharyya coeffcient) from all 
#             # iterations
#             key_opt = list(res.keys())[np.argmax(p_coeff_opt)]
#             res_opt = res[key_opt]
            
#             # The current bandwidth is a filtered version of this iteration's 
#             # optimal bandwidth and previous bandwidths
#             # The filtering takes place in the setter function
#             self.b_width = res_opt['target_region']['b_width'].item()
                    
#             # Now the target region must be scaled according to the filtered 
#             # bandwidth!
#             # To this end take the original target region, shift it according
#             # to the mean shift vector from the best iteration and then scale
#             # it according to the filtered bandwidth
    
            
#             _,res_opt['target_region'] = \
#                 self._update_target_region(target_region,
#                                            scale=self.b_width,
#                                            y=res_opt['meanShift'])
#         else:
#             res_opt = None
            
#         return res_opt
        
#     def _epanechnikov_kernel(self,x,h,**kwargs):
        
#         derivative = kwargs.pop('derivative',False)
        
#         # dimension
#         d = 2
        
#         # volume of the unit d-dimensional sphere
#         c = np.pi
        
#         # Calculate norm of x
#         x = np.linalg.norm(x) / h
        
#         # Check if kernel or derivative of kernel is to be computed
#         if not derivative:
        
#             if abs(x) >= 1:
#                 kE = 0
#             else:
#                 kE = ((0.5*(d+2))/c)  * (1-x)
#                 kE = 1/h * kE
            
#         else:
#             if abs(x) >= 1:
#                 kE = 0
#             else:
#                 kE = -((0.5*(d+2))/c)
            
        
#         return kE
    
    
#     def _bhattacharyya_coefficient(self,q,p):
        
#         bcoeff = []
        
#         for m in q['model'].keys():
            
#             bcoeff.append(np.sqrt(q['model'][m]*p['model'][m]))
            
#         return sum(bcoeff)
    
#     def _calc_weights(self,p:dict,q:dict):
#         """
#         Calculates the weights assigned to each pixel x_i 
#         in the candidate model
        
#         w_i = sum_u [(q_u/p_u)^(1/2) d(b(x_i)) - u]

#         Parameters
#         ----------
#         p : dict
#             Target candidate, i.e. the candidate model whose similarity 
#             to the reference model is evaluated.
#         q : dict
#             Target representation, i.e. the reference model to localize in the
#             frame.

#         Returns
#         -------
#         None.

#         """
        
#         # Get bandwidth from candidate model
#         h = p['b_width']
        
#         # Iniitalize a dictionary for the weights
#         w = {}
        
#         # Loop over the bin_map of the candidate model, which assigns each
#         # pixel to a bin in the histogram which represents the distribution
#         # of the pixels in the target
        
#         for pix in range(len(p['bin_map'])):
            
#             # Get the bin in which the pixel falls
#             b = p['bin_map'][pix]
            
#             # the weight for that pixel is the root of the quotient of 
#             # the probabilities in candidate and target model
            
#             # handle the case of zero division by replacing zero entries
#             # in the candidate model with infinitiy, to get a weight of zero
#             if p['model'][b] == 0:
#                 p_subst = np.inf
#             else:
#                 p_subst = p['model'][b]
            
#             # Calculate the weight w = (q/p)^(1/2)*-g(x)
            
#             # Limit the quotient v = (q/u) to a maximum of 1. Otherwise pixels that 
#             # have a very low probability in the candidate might get a very large 
#             # weight!
#             v = min(q['model'][b] / p_subst,1) 
            
#             # Evaluate derivative of kernel at that pixel
#             g = - self._epanechnikov_kernel(p['x'][pix], h, derivative=True)
            
#             # The weight for this pixel becomes
#             w[pix] = np.sqrt(v) * g
                   
#         return w 
    
#     def _calc_weighted_mean(self,p:dict,w:dict):
#         """
#         Calculate the weighted mean (i.e. the mean shift vector) given the 
#         pixels in the candidate region and its weights

#         Parameters
#         ----------
#         p : dict
#             Target candidate, i.e. the candidate model whose similarity 
#             to the reference model is evaluated.
#         w : dict
#             Weights assigned to each pixel x_i in the candidate region

#         Returns
#         -------
#         None.

#         """
        
#         # Get normalization constants
#         hx = p['norm_const'][0]
#         hy = p['norm_const'][1]
        
#         # Convert dictionary of weights to a numpy array
#         w = np.array(list(w.values())).reshape((-1,1))
        
#         # Calculate weighted mean (add small number to avoid divison by zero)
#         y = sum(p['x'] * w)  / (sum(w)+1E-6)
        
#         # Rescale mean vector 
#         y = y * [hx,hy]
        
#         # Round to full pixels
#         y = np.round(y)
        
#         return y 
    
#     def _calc_bwidth(self,bbox):
#         # The bandwidth is always relative to the initial bandwidth 1, which
#         # is the bandwidth of the target model estimated at the start of 
#         # the track
        
#         # Get normalization constants of target model. 
#         ratio_q = self.q_model['norm_const']
        
#         # Diagonal of target region
#         d_q = np.sqrt((2*ratio_q[0])**2 + (2*ratio_q[1])**2)
        
#         # Get the bounding box coordinates of candidate region
#         xtl = bbox['xtl'].item()
#         ytl = bbox['ytl'].item()
#         xbr = bbox['xbr'].item()
#         ybr = bbox['ybr'].item()
        
#         # get candidate region width, height and diagonal
#         width = xbr - xtl
#         height = ybr - ytl
#         d = np.sqrt(width**2 + height**2) 
        
#         return d/d_q
    
#     def _propose_target_regions(self,bbox):
        
#         # Reset the index to differentiate between different proposed regions
#         bbox = bbox.reset_index()
        
#         # Get normalization constants of target model. They represent the
#         # original aspect ratio, which should not be distorted too much
#         ratio_q = self.q_model['norm_const']
        
#         # Diagonal of target region
#         d_q = np.sqrt((2*ratio_q[0])**2 + (2*ratio_q[1])**2)
        
#         # Get the bounding box coordinates of candidate region
#         xtl = bbox['xtl'].item()
#         ytl = bbox['ytl'].item()
#         xbr = bbox['xbr'].item()
#         ybr = bbox['ybr'].item()
        
#         # get candidate region width, height and diagonal
#         width = xbr - xtl
#         height = ybr - ytl
#         d = np.sqrt(width**2 + height**2) 
        
#         # Calculate bandwidth of candidate region
#         bbox['b_width'] = d/d_q
        
        
#         # Now create a smaller box, an unaltered box and a larger box
#         delta_d = [-2,0,2]
        
#         # Empty list for proposed boxes
#         prop_bboxes = []
        
#         for dd in delta_d:
        
#             # Create a working copy of the bounding box
#             bbox_d = bbox.copy()
            
#             # Assign a unique index
#             d_index = pd.Index(data=[dd],name='delta_d')
#             bbox_d.index = d_index
            
#             # Calculate the diagonal and add delta_d. Then calculate how width
#             # and height of the box should be adapted to yield a box with 
#             # the increased diagnonal while keeping the aspect ratio as well as
#             # possible
#             d_s = np.round(d) + dd
            
#             try:
#                 x_shift = int(np.round(((2*ratio_q[0]/d) * d_s) - width)) // 2
#                 y_shift = int(np.round(((2*ratio_q[1]/d) * d_s) - height)) // 2
                
#             except:
#                 print('bug')
#             # Calculate the new coordinates
#             bbox_d['xtl'] = xtl - x_shift
#             bbox_d['ytl'] = ytl - y_shift
#             bbox_d['xbr'] = xbr + x_shift
#             bbox_d['ybr'] = ybr + y_shift
            
#             # Calculate the bandwidth associated with the altered bounding box
#             # Due to the rounding involved, this can't be pre-calculated
#             width_d = bbox_d['xbr'].item() - bbox_d['xtl'].item()
#             height_d = bbox_d['ybr'].item() - bbox_d['ytl'].item()
            
#             d_s = np.sqrt(width_d**2 + height_d**2)
            
#             # The bandwidth of the target is defined as 1. Normalizing the
#             # diagonal of the altered box w.r.t. the diagonal of the query
#             # box, yields the bandwidth of the altered box
#             bbox_d['b_width'] = d_s/d_q

#             # Append proposed box
#             prop_bboxes.append(bbox_d)
            
#         # In the end concatenate to one dataframe 
#         prop_bboxes = pd.concat(prop_bboxes)
#         # prop_bboxes = prop_bboxes.reset_index(drop=True)
        
#         # Throw out all boxes that are outside the frame or 
#         # not within the specified limits
#         prop_bboxes = self._return_valid_bboxes(prop_bboxes)
               
#         # It can occur, that there are no valid bboxes. In that case, return 
#         # the original target region in the approprate format
#         if len(prop_bboxes)==0:
#             d_index = pd.Index(data=[0],name='delta_d')
#             bbox.index = d_index
#             prop_bboxes.loc[0] = bbox.loc[0]
        
#         return prop_bboxes
        
#     def _update_target_region(self,bbox:pd.DataFrame,**kwargs:dict):
#         """
        

#         Parameters
#         ----------
#         bbox : pd.DataFrame
#             DESCRIPTION.
#         **kwargs : dict
#             DESCRIPTION.

#         Returns
#         -------
#         success : TYPE
#             DESCRIPTION.
#         bbox : TYPE
#             DESCRIPTION.
#         ratio : tuple
#             width to height ratio (w,h)

#         """
        
        
#         # Mean shift vector
#         y = kwargs.pop('y',None)
        
#         # Scale factor to apply to the box
#         scale = kwargs.pop('scale',None)
        
#         # Aspect ratio of the original box (target model)
#         ratio_q = self.q_model['norm_const']
    
    
#         # Initialize variable that shows if shifting and scaling of region
#         # was successfull or not
#         success = True

#         # Get the bounding box coordinates
#         xtl = bbox['xtl'].item()
#         ytl = bbox['ytl'].item()
#         xbr = bbox['xbr'].item()
#         ybr = bbox['ybr'].item()
        
#         # get original width and height
#         width = xbr - xtl
#         height = ybr - ytl
        
        
#         # Shift box if mean shift is provided.
#         if y is not None:   
        
#             # Shift all x coordinates by y[0] and all y coordinates by y[1]
#             xtl = max(0,int(xtl + y[0]))
#             ytl = max(0,int(ytl + y[1]))
#             xbr = min(int(xbr + y[0]),self.width)
#             ybr = min(int(ybr + y[1]),self.height)
        
#             # Check if box still has the same dimensions, if not restore
#             # old dimensions (i.e. box is not shifted)
#             if (xbr - xtl) < width:
#                 xtl = bbox['xtl'].item()
#                 xbr = bbox['xbr'].item()
            
#             if (ybr - ytl) < height:
#                 ytl = bbox['ytl'].item()
#                 ybr = bbox['ybr'].item()
            
#         # Scale box if scale is provided
#         if scale is not None:
                        
#             # Calculate diagonal
#             d = np.sqrt(width**2 + height**2)
            
#             # Scale diagonal
#             d_s = d*scale
            
#             # Given the scaled diagonal and the aspect ration 
#             # calculate the new width and height
#             x_shift = int(np.round(((2*ratio_q[0]/d) * d_s) - width)) // 2
#             y_shift = int(np.round(((2*ratio_q[1]/d) * d_s) - height)) // 2

            
#             # Calculate the coordinates
#             xtl = xtl - x_shift
#             ytl = ytl - y_shift
#             xbr = xbr + x_shift
#             ybr = ybr + y_shift
        
#         # Check if scaled box is out of frame
#         out_box = (xtl<0) or (ytl<0) or (xbr>self.width) or (ybr>self.height)
        
#         # Make sure that scaling did not produce a box smaller than the 
#         # minimal size
#         # Calculate size of scaled box
#         width_d = (xbr - xtl)
#         height_d = (ybr - ytl)
        
#         if width_d<self.minsize or height_d<self.minsize or out_box:
#             success = False
            
#         else:
#             # Update bounding box coordinates
#             bbox['xtl'] = xtl
#             bbox['ytl'] = ytl
#             bbox['xbr'] = xbr
#             bbox['ybr'] = ybr
            
#             # Update bandwidth of box according to actual scaling applied
#             d_s = np.sqrt(width_d**2 + height_d**2)
#             d_q = np.sqrt((2*ratio_q[0])**2 + (2*ratio_q[1])**2)
            
#             bbox['b_width'] = d_s/d_q
        
#         return success,bbox
        
        
#     def _return_valid_bboxes(self,bboxes):
        
#         # Throw out all boxes that are outside the frame
#         idx_drop = bboxes.loc[(bboxes['xtl']<0) | \
#                                (bboxes['ytl']<0) | \
#                                (bboxes['xbr']>self.width) | \
#                                (bboxes['ybr']>self.height)].index   
            
#         bboxes = bboxes.drop(index = idx_drop)
        
#         # Throw out all boxes that are not within the specified limits
#         width = bboxes['xbr'] - bboxes['xtl']
#         height = bboxes['ybr'] - bboxes['ytl']
        
#         idx_drop = bboxes.loc[(width<self.minsize) | \
#                               (height<self.minsize)].index
            
#         bboxes = bboxes.drop(index = idx_drop)
        
#         return bboxes

# class meanShift_ar(meanShift_ar_adaptive):
    
#     def __init__(self, width, height, **kwargs):
#         super().__init__(width,height,**kwargs)
    
#     def _propose_target_regions(self,bbox):
        
#         d_index = pd.Index(data=[0],name='delta_d')
#         bbox.index = d_index
#         prop_bboxes = bbox.loc[[0]].copy()
#         prop_bboxes['b_width'] = 1
        
        
#         return prop_bboxes