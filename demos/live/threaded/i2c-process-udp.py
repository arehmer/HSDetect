# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 09:14:10 2024

@author: rehmer
"""

import pickle as pkl
from queue import Queue
import time
from pathlib import Path
import argparse  

from hspy.ipc.threads import UDP_PickleServer, I2C_Client
from hspy.drivers.i2c import I2C_HTPA32x32d
from hsod.ipc.threads import ProcessingThread
from hspy.LuT import LuT

from hsod.cv.detectors import build_detector


# %% Create an argument parser to enable passing argument from the
# command line to this script
arg_parser = argparse.ArgumentParser(prog = 'I2C_ProcessRecord.py',
                                     description="Connects to a HTPA device via I2C, applies a processing engine and writes the result to a directory.")

# %% Add arguments using '--key' style
arg_parser.add_argument("--bus",
                        dest = "bus",
                        type=int,
                        required=True,
                        help="I2C bus that HTPA device is connected to.")

arg_parser.add_argument("--target_ip",
                        dest="target_ip",
                        type=str,
                        required=True,
                        help="IP address of the peer running UDP_PickleClient.")

arg_parser.add_argument("--target_port",
                        dest="target_port",
                        type=int,
                        required=True,
                        help="Port the peer's UDP_PickleClient is listening on.")

# %% Parse arguments
args = arg_parser.parse_args()
bus = args.bus
target_ip = args.target_ip
target_port = args.target_port


# %% Main loop
if __name__ == '__main__':
    
    
    #%% Create buffers for communication between threads
    i2c_buffer = Queue(maxsize=1)           # Buffer that i2c thread puts data in
    processor_buffer = Queue(maxsize=1)     # Buffer that processing thread puts data in
    
    #%% Create instance of i2c driver
    # __init__ already opens the bus and calls init(), so the driver is
    # ready to acquire frames as soon as it is constructed
    i2c_driver = I2C_HTPA32x32d(bus)
    
    # Load and assign the LuT
    lut154 = LuT()
    lut154.from_csv(Path.cwd() / 'lut/32x32dL1k7.csv', offset=0)
    i2c_driver.LuT = lut154
    
    # %% Create I2C_Client thread, which repeatedly calls
    # i2c_driver.acquire_frame() and puts the result into i2c_buffer
    i2c_thread = I2C_Client(i2c_driver = i2c_driver,
                            write_buffer = i2c_buffer)
       
    # %% Create ProcessorThread
    # detector_dict = pkl.load(open('ProposalDetectorParams.dict','rb'))
    detector_dict = pkl.load(open(Path.cwd() / 'models' / 'OneScaleDetectorParams.dict','rb'))
    
    detector = build_detector(**detector_dict)
    processor_thread = ProcessingThread(name = 'processor_thread',
                                        detector = detector,
                                        write_buffer = processor_buffer,
                                        read_buffer = i2c_buffer)

        
    #%% Create instance of UDP_PickleServer thread
    server_thread = UDP_PickleServer(name='udp_pickle_server',
                                     read_buffer=processor_buffer,
                                     target_ip=target_ip,
                                     target_port=target_port)
    
if __name__ == '__main__':
     
    print('Starting threads...')  
    i2c_thread.start()
    processor_thread.start()
    server_thread.start()
    print('Threads started.')  
    
    try:
        # Keep the main thread alive and responsive without wastin
        # too many ressources, waiting for Ctrl+C
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping threads...")
    finally:
        # Stop in reverse order regardless of how we exit
        server_thread.stop()
        processor_thread.stop()
        i2c_thread.stop()
        
        i2c_thread.join()
        processor_thread.join()
        server_thread.join()
        
        print("All threads stopped.")