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

from hspy.ipc.processes_base import SharedMemoryQueue
from hspy.ipc.processes import UDP_PickleClientProcess, ImshowProcess

# %% Create an argument parser to enable passing arguments from the
# command line to this script
arg_parser = argparse.ArgumentParser(prog='udp-imshow-mp.py',
                                     description="Receives dicts containing a numpy array and a pandas DataFrame sent by a UDP_PickleServer(Process) peer -- multiprocessing version.")

# %% Add arguments using '--key' style
arg_parser.add_argument("--listen_ip",
                        dest="listen_ip",
                        type=str,
                        required=False,
                        default='0.0.0.0',
                        help="Local IP address to listen on.")

arg_parser.add_argument("--listen_port",
                        dest="listen_port",
                        type=int,
                        required=True,
                        help="Local port to listen on. Must match --target_port passed to the peer's i2c-process-udp-mp.py.")

# %% Parse arguments
args = arg_parser.parse_args()
listen_ip = args.listen_ip
listen_port = args.listen_port

# %% Main
#
# NOTE: as in i2c-process-udp-mp.py, all Process / SharedMemoryQueue
# construction must live inside a single `if __name__ == '__main__':`
# guard for multiprocessing to be safe across start methods.
if __name__ == '__main__':

    print('Initializing objects...')

    # Buffer that UDP_PickleClientProcess writes received data into
    recv_buffer = SharedMemoryQueue(maxsize=1, slot_bytes=4 * 1024 * 1024)

    # Create instance of UDP_PickleClientProcess
    client_process = UDP_PickleClientProcess(name='udp_pickle_client_process',
                                             write_buffer=recv_buffer,
                                             listen_ip=listen_ip,
                                             listen_port=listen_port)

    #%% Create instance of ImshowProcess for plotting using cv2.imshow
    plot_process = ImshowProcess(name='imshow_process',
                                 read_buffer=recv_buffer,
                                 ArrayType=10)

    print('Starting processes...')
    client_process.start()
    plot_process.start()

    # NOTE: unlike UDP_PickleClient.listen_addr (a plain property that
    # works immediately because the thread shares the parent's memory),
    # the socket here only exists inside the child process once it has
    # actually bound -- get_listen_addr() blocks briefly and waits for
    # the child to report back, rather than reading a value that would
    # not exist yet in the parent.
    print(f'Listening on {client_process.get_listen_addr()}...')

    try:
        # Keep running, receiving and displaying frames until Ctrl+C
        while True:
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nStopping processes...")
    finally:
        plot_process.stop()
        client_process.stop()

        plot_process.join()
        client_process.join()

        recv_buffer.close()

        print("Processes stopped.")

    print('End')
