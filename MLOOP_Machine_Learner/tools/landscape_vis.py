#!/usr/bin/env python
import logging
import mloop.utilities as mlu
import mloop.visualizations as mlv
import argparse

parser = argparse.ArgumentParser(
    description='Plot cross sections of the predicted landscape, and optionally upload them via plotly. Must be run from the same directory as M-LOOP was run.')
parser.add_argument("controller_filename")
parser.add_argument("learner_filename")
parser.add_argument("learner_type")
parser.add_argument("-u", "--upload", action="store_true",
                    help="upload plots to the interwebs")
args = parser.parse_args()


mlu.config_logger(log_filename=None, console_log_level=logging.DEBUG)

mlv.show_all_default_visualizations_from_archive(
    args.controller_filename, args.learner_filename, args.learner_type, upload_cross_sections=args.upload)
