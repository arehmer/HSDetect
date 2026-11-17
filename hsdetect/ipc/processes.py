import time
import traceback
from pathlib import Path

from hspy.tparray import TPArray

from hsdetect import build_engine
from hsdetect.engine_io import find_engine_params_file, load_engine_params

from hspy.ipc.processes import RWProcess_R1



class CVProcess(RWProcess_R1):
    """
    Multiprocessing version of ProposalDetectorThread (ProcessorThread.py).
    Reads frames from read_buffer, runs object detection, writes the
    result (original data + detections) into write_buffer.
    """

    def __init__(self,
                 name: str,
                 model_dict: dict,
                 write_buffer,
                 read_buffer,
                 **kwargs):
        """
        Parameters
        ----------
        model_dict : dict
            Kwargs passed to build_detector(**model_dict) or 
            build_tracktor(**model_dict )inside
            _setup(). The detector/tracktor is built inside the child process
            rather than being constructed in the parent and pickled
            across, since detector/model objects frequently hold
            resources (loaded weights, a GPU/inference context, ...)
            that either aren't picklable or shouldn't be duplicated
            across a process boundary.
        write_buffer, read_buffer : Queue / SharedMemoryQueue

        Returns
        -------
        None.

        """

        self.model_dict = model_dict

        super().__init__(name=name,
                         read_buffer=read_buffer,
                         write_buffer=write_buffer,
                         **kwargs)

    def _setup(self):
        
        # Check if the provided model is a detector or tracker by name
        # and build the model (load weights / params, allocate whatever
        # runtime context it needs) here, inside the child process
        name = self.model_dict['name']  
        
        # Imported here, so that the other processes (e.g. InferenceProcess)
        # do not load the classical detectors and their dependencies
        import hsdetect.cv.detectors
        import hsdetect.cv.tracktors
        from hsdetect.cv.detectors import build_detector
        from hsdetect.cv.tracktors import build_tracktor

        if name in hsdetect.cv.detectors.get_classes():
            print(f'Provided model {name} identified as detector.')
            self.model = build_detector(**self.model_dict)
        elif name in hsdetect.cv.tracktors.get_classes():
            print(f'Provided model {name} identified as tracktor.')
            self.model = build_tracktor(**self.model_dict)
        else:
            print(f'Provided model {name} could not be identified.')
            return None

        self.tparray = TPArray(SensorType=self.model.SensorType)

        self.num_pix = len(self.tparray._pix)

        self.t0 = time.time()

    def _target(self):

        # Get result from upstream process
        data = self.read_buffer.get()

        # Check success flag of upstream process
        if data['success'] is True:

            if 'pix_dK' in data.keys():
                frame = data['pix_dK']
            elif 'frame' in data.keys():
                frame = data['frame']

            # Reshape if not the proper size
            if frame.ndim == 1:
                frame = frame[0:self.num_pix]
                frame = frame.reshape(self.tparray._npsize)

            try:
                data_processed = self.model.process({'frame': frame,
                                                     'idx': data['image_id']},
                                                    update_bg = True)
                data.update(data_processed)

            except Exception as e:
                print(f"[{self.name}] {e}")

                # Set success flag to False in case of failure
                data['success'] = False

        else:
            # If upstream process failed, set success flag to False
            data['success'] = False

        return data

    def stop(self):
        self._exit.set()
        
        
class InferenceProcess(RWProcess_R1):
    """
    Class for running a hsdetect.cv.inference engine as a process..
    Reads frames from read_buffer, runs object detection, writes the
    result (original data + detections) into write_buffer.
    """
    
    def __init__(self,
                 name: str,
                 inference_engine_dir: Path,
                 write_buffer,
                 read_buffer,
                 **kwargs):
        
        self.inference_engine_dir = Path(inference_engine_dir)
        self.inference_engine = None
        self._internal_frame_idx = 0
        
        super().__init__(name=name,
                         read_buffer=read_buffer,
                         write_buffer = write_buffer,
                         **kwargs)
        
    def _setup(self,):
        
        # Build the model from the parameter file (.json) in the directory
        # given. Model paths in it are relative to the directory, see
        # hsdetect.engine_io.load_engine_params().
        try:
            file = find_engine_params_file(self.inference_engine_dir)
            inf_engine_params = load_engine_params(file)
            self.inference_engine = build_engine(**inf_engine_params)
            print(f'Built inference engine from {file.name} with engines: '
                  f'{self.inference_engine.engine_names}.')
        except Exception:
            print(f'[{self.name}] Failed to build inference engine from '
                  f'{self.inference_engine_dir}:')
            traceback.print_exc()
                
        if self.inference_engine is None:
            raise RuntimeError('Was not able to build inference engine from ' + \
                               f'{self.inference_engine_dir}.')
        
            
    def _target(self):

        # Get result from upstream process
        data = self.read_buffer.get()
        
        
        # If success flag is set, pass frame to inference engine
        if data['success'] == True:
            
            # Get frame 
            frame = data['frame'].reshape((120,160))
            
            try:
                # Pass frame to inference engine
                inference = self.inference_engine(frame)
                
                
                # Add result to dict
                data['inference'] = inference
                
                # Set success flag to True
                data['success'] = True
                
            except Exception as e:
                print(f"[{self.name}] {e}")
                
                # Set success flag to False in case of failure
                data['success'] = False

        return data
