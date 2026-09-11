# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 09:14:10 2024

@author: rehmer
"""

import pickle as pkl
import time
from pathlib import Path
import argparse  

from hspy.ipc.processes_base import SharedMemoryQueue

from hspy.readers import HTPA_UDPReader
from hspy.ipc.processes import ImshowProcess, UDP_ClientProcess
from hsdetect.ipc.processes import CVProcess

# %% Create an argument parser to enable passing argument from the
# command line to this script
arg_parser = argparse.ArgumentParser(prog = 'wifi-process-imshow.py',
                                      description="Connects to a HTPA device via UDP, applies a processing engine and writes the result to a directory.")

# %% Add arguments using '--key' style
arg_parser.add_argument("--bcast",
                        dest = "bcast",
                        type=str,
                        required=True,
                        help="Broadcast Address of the subnet to search for HTPA devices. Format: xxx.xxx.x.255")


# %% Parse arguments
args = arg_parser.parse_args()
bcast_addr = args.bcast


# %% Main 
if __name__ == '__main__':
    
    #%% Create shared-memory-backed buffers for communication between
    # processes (replaces the plain queue.Queue buffers used between
    # threads)
    udp_buffer = SharedMemoryQueue(maxsize=4, slot_bytes=4 * 1024 * 1024)
    processor_buffer = SharedMemoryQueue(maxsize=4, slot_bytes=4 * 1024 * 1024)
    
    # %% Create instance of UDP reader
    udp_reader = HTPA_UDPReader()
    
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
    
    #%% Create instance of UDP thread
    udp_process = UDP_ClientProcess(udp_reader_cls = HTPA_UDPReader,
                                    udp_reader_kwargs = {'ArrayType':ArrayType},
                                    DevID = DevID,
                                    Bcast_Addr = bcast_addr,
                                    write_buffer = udp_buffer)
    
    # %% Create ProcessorThread
    # detector_dict = pkl.load(open('ProposalDetectorParams.dict','rb'))
    model_dict = pkl.load(open(Path.cwd() / 'models' / 'HSv01_HTPA32x32d_L1K7_0k8_ang90_24_08_2026.dict','rb'))

    processor_process = CVProcess(name = 'cv_process',
                                  model_dict = model_dict,
                                  write_buffer = processor_buffer,
                                  read_buffer = udp_buffer,
                                  profile = True,
                                  profile_every= 50)
    
    #%% Create instance of ImshowProcess for plotting using cv2.imshow
    plot_process = ImshowProcess(name='imshow_process',
                                 read_buffer=processor_buffer,
                                 ArrayType=10)


if __name__ == '__main__':
    
    print('Starting threads...')  
    udp_process.start()
    processor_process.start()
    plot_process.start()
    print('Threads started.')  
    
    try:
        # Keep the main thread alive, waiting for Ctrl+C
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping threads...")
    finally:
        # Stop in reverse order regardless of how we exit
        plot_process.stop()
        processor_process.stop()
        udp_process.stop()
        
        print("All threads stopped.")
