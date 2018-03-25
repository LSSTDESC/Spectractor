from spectractor import *
from logbook import *
import collections

            
if __name__ == "__main__":
    import commands, string, re, time, os
    from argparse import ArgumentParser
    import glob

    parser = ArgumentParser()
    parser.add_argument(dest="input", metavar='path',default=["notebooks/fits/trim_20170605_007.fits"],help="Input fits file name. It can be a list separated by spaces, or it can use * as wildcard.",nargs='*')
    parser.add_argument("-d", "--debug", dest="debug",action="store_true",
                      help="Enter debug mode (more verbose and plots).",default=False)
    parser.add_argument("-v", "--verbose", dest="verbose",action="store_true",
                      help="Enter verbose (print more stuff).",default=False)
    parser.add_argument("-o", "--output_directory", dest="output_directory", default="test/",
                      help="Write results in given output directory (default: ./tests/).")
    args = parser.parse_args()

    parameters.VERBOSE = args.verbose
    if args.debug:
        parameters.DEBUG = True
        parameters.VERBOSE = True

    filenames = args.input
    
    logbook = LogBook()
    for filename in filenames:
        tag = filename.split('/')[-1]
        target, xpos, ypos = logbook.search_for_image(tag)
        if target is None or xpos is None or ypos is None: continue
        Spectractor(filename,args.output_directory,[xpos,ypos],target)
