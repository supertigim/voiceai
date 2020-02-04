# -*- coding: utf-8 -*-
import math
import struct

from config import Config as CFG  

def rms(frame):
    count = len(frame)/CFG.S_WIDTH
    format = "%dh"%(count)
    # short is 16 bit int
    shorts = struct.unpack( format, frame )

    sum_squares = 0.0
    for sample in shorts:
        n = sample * CFG.SHORT_NORMALIZE
        sum_squares += n*n
    # compute the rms 
    rms = math.pow(sum_squares/count,0.5);
    return rms * 1000

# end of file