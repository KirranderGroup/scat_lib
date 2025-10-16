"""
Settings for the scat_lib package.

"""
import os
import pathlib

#Defaults to executable in scat_lib/bin
SCAT_EXE_PATH = os.path.join(pathlib.Path(__file__).parent, 'bin', 'scat')

