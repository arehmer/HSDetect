# -*- coding: utf-8 -*-
"""
Posture counterpart of udp-process-imshow.py.

Demonstrates UDP_ClientProcess + PostureInferenceProcess +
ImshowPostureProcess: listens for frames streamed by an HTPA device,
classifies the posture in every frame with the PostureNet CNN, and
visualizes the background-removed frame with the result burned in --
each stage running in its own OS process, connected via
SharedMemoryQueue.

Unlike udp-process-imshow.py there is no engine chain: PostureNetEngine
does its own background removal, so the raw frame goes straight into it.


IMPORTANT -- the first frames build the background
==================================================
PostureNetEngine estimates its background from the first --bg_frames
frames (5 by default) and then freezes it, exactly as
CNNDetection/backgroundRemover.py does for a recording. Those frames
must show the EMPTY room: anybody in the field of view while the window
still says "estimating background" is subtracted away for the rest of
the run. Restart the script, or call reset() on the engine, to estimate
it again.

The checkpoint is the 160x120 PostureNet, so this demo expects an
HTPA 160x120. On a different array the engine refuses the frame and says
so rather than resampling it.


Usage Examples
==============

Windows PowerShell:
python ./udp-posture-imshow.py --bcast 192.168.1.255

Linux shell (bash):
python ./udp-posture-imshow.py --bcast 192.168.1.255


Command-line Arguments
======================
--bcast : str (required)
    Broadcast address of the subnet to search for HTPA devices.
--model : str (optional)
    PostureNet checkpoint to run.
--bg_frames : int (optional, default 5)
    Leading frames averaged into the background estimate.
--device : str (optional, default 'auto')
    Torch device: 'auto', 'cpu', 'cuda'.
--no-imshow : flag (optional)
    Disables cv2.imshow().
"""

print('Loading libraries...')

import argparse
from pathlib import Path

from hspy.readers import HTPA_UDPReader
from hspy.tparray import SensorTypes
from hspy.ipc.processes_base import SharedMemoryQueue
from hspy.ipc.processes import UDP_ClientProcess
from hspy.ipc.threads_base import RThread

from hsdetect.ipc.processes import (PostureInferenceProcess,
                                    ImshowPostureProcess)


SCRIPT_DIR = Path(__file__).resolve().parent

# Default checkpoint: the 160x120 PostureNet of the CNNDetection project, as
# copied into this demo folder. Resolved against the script's own directory
# rather than the working directory, so the demo runs from anywhere.
DEFAULT_MODEL = SCRIPT_DIR / 'models' / 'fdPostureCNN' / 'best.pt'


# %% Create an argument parser to enable passing arguments from the
# command line to this script
arg_parser = argparse.ArgumentParser(prog='udp-posture-imshow.py',
                                     description="Classifies the posture in every frame streamed by an HTPA device and visualizes the result -- multiprocessing version.")

# %% Add arguments using '--key' style
arg_parser.add_argument("--bcast",
                        dest = "bcast",
                        type=str,
                        required=True,
                        help="Broadcast Address of the subnet to search for HTPA devices. Format: xxx.xxx.x.255")

arg_parser.add_argument("--model",
                        dest = "model",
                        type = str,
                        required = False,
                        default = str(DEFAULT_MODEL),
                        help = f"PostureNet checkpoint (.pt) to run. Default: {DEFAULT_MODEL}")

arg_parser.add_argument("--bg_frames",
                        dest = "bg_frames",
                        type = int,
                        required = False,
                        default = 5,
                        help = "Leading frames averaged into the background estimate. They must show the empty room.")

arg_parser.add_argument("--device",
                        dest = "device",
                        type = str,
                        required = False,
                        default = 'auto',
                        help = "Torch device: 'auto', 'cpu' or 'cuda'.")

arg_parser.add_argument("--no-imshow",
                        dest = "imshow",
                        action = "store_false",
                        required = False,
                        help = "Flag disabling cv2.imshow()")

# %% Parse arguments
args = arg_parser.parse_args()
bcast_addr = args.bcast
imshow = args.imshow

# %% Main
#
# NOTE: as in udp-process-imshow.py, all Process / SharedMemoryQueue
# construction must live inside a single `if __name__ == '__main__':`
# guard for multiprocessing to be safe across start methods.
if __name__ == '__main__':

    # %% Create instance of UDP reader
    udp_reader = HTPA_UDPReader()

    print('Initializing objects...')

    # %% Broadcast looking for devices
    devices = udp_reader.broadcast(bcast_addr)

    # If no devices found, end script
    if len( udp_reader.devices.index) == 0:
        raise ValueError('No devices found.')

    # Get user input on which device to bind
    while True:
        try:
            DevID = int(input('Enter the device number of the HTPA device to bind: ' ))
            if DevID in udp_reader.devices.index:
                break
            else:
                print(f'HTPA device with ID {DevID} not discovered.')
        except ValueError:
            print("Please enter a valid integer.")

    # Get the ArrayType of the selected device
    ArrayType = devices.loc[DevID,'Arraytype']

    # Buffer that UDP_ClientProcess writes received data into
    udp_buffer = SharedMemoryQueue(maxsize=1, slot_bytes=4 * 1024 * 1024)
    posture_buffer = SharedMemoryQueue(maxsize=1, slot_bytes=4 * 1024 * 1024)

    # Create instance of UDP_ClientProcess
    client_process = UDP_ClientProcess(udp_reader_cls = HTPA_UDPReader,
                                       udp_reader_kwargs = {'ArrayType': ArrayType},
                                       write_buffer=udp_buffer,
                                       name='udp_client_process',
                                       DevID = DevID,
                                       Bcast_Addr = bcast_addr)

    # Parameters of the posture engine. Passed as a dict rather than as a
    # built engine: the engine holds a loaded torch model, which is built
    # inside the child process instead of being pickled across.
    engine_dict = {'name': 'PostureNetEngine',
                   'SensorType': SensorTypes['HTPA160x120DR1_L3K95_0K8'],
                   'model_path': args.model,
                   'bg_frames': args.bg_frames,
                   'device': args.device}

    # Create instance of the PostureInferenceProcess
    posture_process = PostureInferenceProcess(name = 'posture_process',
                                              engine_dict = engine_dict,
                                              ArrayType = ArrayType,
                                              read_buffer = udp_buffer,
                                              write_buffer = posture_buffer)

    # Distinguish between the cases of plotting the sensor stream or not
    if imshow:
        # Create instance of the posture viewer, plotting with cv2.imshow
        plot_process = ImshowPostureProcess(ArrayType = ArrayType,
                                            name = 'imshow_posture_process',
                                            read_buffer = posture_buffer)

    elif not imshow:
        # Create a dummy read thread, that only empties the queue and releases
        # the lock on the writing condition
        def dummy_target():
            posture_buffer.get()  # drain one item from the queue each cycle

        plot_process = RThread(target = dummy_target,
                               read_buffer = posture_buffer)

    print('Starting processes...')
    print(f'Building the background from the first {args.bg_frames} frames - '
          'keep the room empty.')
    client_process.start()
    posture_process.start()
    plot_process.start()
    print('Processes started.')


    try:
        # Keep the main process alive and responsive without wasting
        # too many resources, waiting for Enter
        input('Press Enter to stop...\n')
        print("Stopping processes...")

    except KeyboardInterrupt:
        # Ctrl+C still works as a fallback; EOFError if stdin is closed
        print("\nStopping processes...")
    finally:
        plot_process.stop()
        posture_process.stop()
        client_process.stop()

        plot_process.join()
        posture_process.join()
        client_process.join()

        udp_buffer.close()
        posture_buffer.close()
        print("All processes stopped.")
