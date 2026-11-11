# -*- coding: utf-8 -*-
"""
Multiprocessing counterpart of udp-imshow.py.

Demonstrates UDP_PickleClientProcess + ImshowProcess: listens for
messages sent by a peer running UDP_PickleServer(Process) (see
i2c-process-udp-mp.py), reassembles and unpickles them, and visualizes
the received frame + detections with OpenCV -- each stage running in its
own OS process, connected via SharedMemoryQueue.

Run this script and i2c-process-udp-mp.py on the same machine (using
127.0.0.1) or on two different machines on the same network.


Usage Examples
==============

Windows PowerShell:
python ./udp-imshow-mp.py --listen_ip 127.0.0.1 --listen_port 5005

Linux shell (bash):
python ./udp-imshow-mp.py --listen_ip 127.0.0.1 --listen_port 5005


Command-line Arguments
======================
--listen_ip : str (optional, default '0.0.0.0')
    Local IP address to listen on.
--listen_port : int (required)
    Local port to listen on. Must match --target_port passed to the
    peer's i2c-process-udp-mp.py.
"""

print('Loading libraries...')

import time
import argparse
from pathlib import Path

from hspy.readers import HTPA_UDPReader
from hspy.ipc.processes_base import SharedMemoryQueue
from hspy.ipc.processes import UDP_ClientProcess, ImshowProcess
from hsdetect.ipc.processes import InferenceProcess
from hspy.ipc.threads_base import RThread


SCRIPT_DIR = Path(__file__).resolve().parent

# %% Create an argument parser to enable passing arguments from the
# command line to this script
arg_parser = argparse.ArgumentParser(prog='udp-process-imshow.py',
                                     description="Receives dicts containing a numpy array and a pandas DataFrame sent by a UDP_PickleServer(Process) peer -- multiprocessing version.")

# %% Add arguments using '--key' style
arg_parser.add_argument("--bcast",
                        dest = "bcast",
                        type=str,
                        required=True,
                        help="Broadcast Address of the subnet to search for HTPA devices. Format: xxx.xxx.x.255")

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
# NOTE: as in i2c-process-udp-mp.py, all Process / SharedMemoryQueue
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

    # Buffer that UDP_PickleClientProcess writes received data into
    udp_buffer = SharedMemoryQueue(maxsize=1, slot_bytes=4 * 1024 * 1024)
    process_buffer = SharedMemoryQueue(maxsize=1, slot_bytes=4 * 1024 * 1024)

    # Create instance of UDP_PickleClientProcess
    client_process = UDP_ClientProcess(udp_reader_cls = HTPA_UDPReader,
                                       udp_reader_kwargs = {'ArrayType': ArrayType},
                                       write_buffer=udp_buffer,
                                       name='udp_client_process',
                                       DevID = DevID,
                                       Bcast_Addr = bcast_addr)
    
    # Create instance of the InferenceProcess
    inf_process = InferenceProcess(name = 'inference_process',
                                   inference_engine_dir = SCRIPT_DIR / 'models' / 'v14_sgd_live',
                                   read_buffer = udp_buffer,
                                   write_buffer = process_buffer)

    # Distinguish between the cases of plotting the sensor stream or not
    if imshow: 
        # Create instance of Imshow thread for plotting using cv2.imshow
        plot_process = ImshowProcess(ArrayType = ArrayType,
                                     name = 'imshow_process',
                                     read_buffer = process_buffer)
            
    elif not imshow:
        # Create a dummy read thread, that only empties the queue and releases 
        # the lock on the writing condition
        def dummy_target():
            udp_buffer.get()  # drain one item from the queue each cycle
        
        plot_process = RThread(target = dummy_target,
                               read_buffer = process_buffer)

    print('Starting processes...')
    client_process.start()
    inf_process.start()
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
        inf_process.stop()
        client_process.stop()

        plot_process.join()
        inf_process.join()
        client_process.join()

        udp_buffer.close()
        process_buffer.close()
        print("All processes stopped.")
