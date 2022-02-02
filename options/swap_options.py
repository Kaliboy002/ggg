import argparse


class SwapOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # paths (data, models, etc...)
        self.parser.add_argument('--arcface_path', type=str,
                                 default="arcface_model/arcface/arc_res50.h5",
                                 help='path to arcface model. Used to extract identity from source.')

        # Video/Image necessary models
        self.parser.add_argument('--retina_path', type=str,
                                 default="retinaface/retinaface_res50.h5",
                                 help='path to retinaface model.')
        self.parser.add_argument('--compare', type=bool,
                                 default=True,
                                 help='If true, concatenates the frame with the manipulated frame')

        self.parser.add_argument('--load', type=int,
                                 default=30,
                                 help='int of number to load checkpoint weights.')
        self.parser.add_argument('--device_id', type=int, default=0,
                                 help='which device to use')

        # logging and checkpointing
        self.parser.add_argument('--log_dir', type=str, default='logs/runs/',
                                 help='logging directory')
        self.parser.add_argument('--log_name', type=str, default='affa_f',
                                 help='name of the run, change this to track several experiments')

        self.parser.add_argument('--chkp_dir', type=str, default='checkpoints/',
                                 help='checkpoint directory (will use same name as log_name!)')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt