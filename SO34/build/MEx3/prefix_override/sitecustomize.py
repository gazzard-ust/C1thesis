import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/c1/Documents/C1_THESIS/SO34/install/MEx3'
