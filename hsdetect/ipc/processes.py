from hspy.tparray import TPArray

import hsdetect
from hsdetect.cv.detectors import build_detector
from hsdetect.cv.tracktors import build_tracktor

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