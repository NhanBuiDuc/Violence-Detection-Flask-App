import os
from typing import Dict

import logging
import numpy as np
import torch
import cv2
import torchvision
from queue import Queue
from Feature_Extractor.models._base.base_extractor import BaseExtractor
from Feature_Extractor.models.i3d.i3d_src.i3d_net import I3D
from Feature_Extractor.models.pwc.extract_pwc import DATASET_to_PWC_CKPT_PATHS
from Feature_Extractor.models.raft.extract_raft import DATASET_to_RAFT_CKPT_PATHS
from Feature_Extractor.models.raft.raft_src.raft import RAFT, InputPadder
from Feature_Extractor.models.transforms import (Clamp, PermuteAndUnsqueeze, PILToTensor,
                               ResizeImproved, ScaleTo1_1, TensorCenterCrop,
                               ToFloat, ToUInt8)
from Feature_Extractor.utils.utils import dp_state_to_normal, show_predictions_on_dataset


class ExtractI3D(BaseExtractor):

    def __init__(self, args) -> None:
        # init the BaseExtractor
        super().__init__(
            feature_type=args.feature_type,
            on_extraction=args.on_extraction,
            tmp_path=args.tmp_path,
            output_path=args.output_path,
            keep_tmp_files=args.keep_tmp_files,
            device=args.device,
        )
        # (Re-)Define arguments for this class
        self.streams = ['rgb']
        self.flow_type = args.flow_type
        self.i3d_classes_num = 400
        self.min_side_size = 256
        self.central_crop_size = 224
        self.extraction_fps = args.extraction_fps
        self.step_size = 64 if args.step_size is None else args.step_size
        self.stack_size = 64 if args.stack_size is None else args.stack_size
        self.resize_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            ResizeImproved(self.min_side_size),
            PILToTensor(),
            ToFloat(),
        ])
        self.i3d_transforms = {
            'rgb': torchvision.transforms.Compose([
                TensorCenterCrop(self.central_crop_size),
                ScaleTo1_1(),
                PermuteAndUnsqueeze()
            ]),
            'flow': torchvision.transforms.Compose([
                TensorCenterCrop(self.central_crop_size),
                Clamp(-20, 20),
                ToUInt8(),
                ScaleTo1_1(),
                PermuteAndUnsqueeze()
            ])
        }
        self.show_pred = args.show_pred
        # self.output_feat_keys = self.streams + ['fps', 'timestamps_ms']
        self.output_feat_keys = self.streams
        self.name2module = self.load_model()
        
    @torch.no_grad()
    def extract_demo(self, video_queue: Queue, extract_event) -> Dict[str, np.ndarray]:
        """The extraction call. Made to clean the forward call a bit.

        Arguments:
            video_path (str): a video path from which to extract features

        Returns:
            Dict[str, np.ndarray]: feature name (e.g. 'fps' or feature_type) to the feature tensor
            """

        logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)

        logging.debug('start extract')
        rgb_stack = []
        feats_dict = {stream: [] for stream in self.streams}

        # sometimes when the target fps is 1 or 2, the first frame of the reencoded video is missing
        # and cap.read returns None but the rest of the frames are ok. timestep is 0.0 for the 2nd frame in
        # this case
        first_frame = True
        padder = None
        stack_counter = 0
        frame = video_queue.get()
        # preprocess the image
        for rgb in frame.rgb:

            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = self.resize_transforms(rgb)
            rgb = rgb.unsqueeze(0)

            if self.flow_type == 'raft' and padder is None:
                padder = InputPadder(rgb.shape)

            rgb_stack.append(rgb)

            # - 1 is used because we need B+1 frames to calculate B frames
            if len(rgb_stack) - 1 == self.stack_size:
                batch_feats_dict = self.run_on_a_stack(rgb_stack)
                for stream in self.streams:
                    feats_dict[stream].extend(batch_feats_dict[stream].tolist())
                    # leaving the elements if step_size < stack_size so they will not be loaded again
                    # if step_size == stack_size one element is left because the flow between the last element
                    # in the prev list and the first element in the current list
                    rgb_stack = rgb_stack[self.step_size:]
                    stack_counter += 1
        if video_queue.empty():
            extract_event.wait()
        frame.i3d = feats_dict['rgb']
        return frame

    @torch.no_grad()
    def extract(self, interval_extract_event, frames_queue, i3d_queue) -> Dict[str, np.ndarray]:
        """The extraction call. Made to clean the forward call a bit.

        Arguments:
            video_path (str): a video path from which to extract features

        Returns:
            Dict[str, np.ndarray]: feature name (e.g. 'fps' or feature_type) to the feature tensor
            """
        torch.cuda.empty_cache()
        logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)

        logging.debug('start extract')
        rgb_stack = []
        feats_dict = {stream: [] for stream in self.streams}

        # sometimes when the target fps is 1 or 2, the first frame of the reencoded video is missing
        # and cap.read returns None but the rest of the frames are ok. timestep is 0.0 for the 2nd frame in
        # this case
        stack_counter = 0

        if(len(frames_queue) > 0):
            current_frame = frames_queue.pop(0)
            # preprocess the image
            for rgb in current_frame.rgb:

                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                rgb = self.resize_transforms(rgb)
                rgb = rgb.unsqueeze(0)

                rgb_stack.append(rgb)

                    # - 1 is used because we need B+1 frames to calculate B frames
                if len(rgb_stack) - 1 == self.stack_size:
                    batch_feats_dict = self.run_on_a_stack(rgb_stack)
                    feats_dict['rgb'].extend(batch_feats_dict['rgb'].tolist())
                    # leaving the elements if step_size < stack_size so they will not be loaded again
                    # if step_size == stack_size one element is left because the flow between the last element
                    # in the prev list and the first element in the current list
                    rgb_stack = rgb_stack[self.step_size:]
                    stack_counter += 1
            current_frame.i3d = feats_dict['rgb']
            i3d_queue.append(current_frame)

        if len(frames_queue) <= 0:
            interval_extract_event.wait()
        return

    def run_on_a_stack(self, rgb_stack) -> Dict[str, torch.Tensor]:
            models = self.name2module['model']
            rgb_stack = torch.cat(rgb_stack).to(self.device)

            batch_feats_dict = {}
            stream_slice = rgb_stack[:-1]
            stream_slice = self.i3d_transforms['rgb'](stream_slice)
            batch_feats_dict['rgb'] = models['rgb'](stream_slice, features=True)

            return batch_feats_dict

    def load_model(self) -> Dict[str, torch.nn.Module]:
        """Defines the models, loads checkpoints, sends them to the device.
        Since I3D is two-stream, it may load a optical flow extraction model as well.

        Returns:
            Dict[str, torch.nn.Module]: model-agnostic dict holding modules for extraction and show_pred
        """
        flow_model_paths = {'pwc': DATASET_to_PWC_CKPT_PATHS['sintel'], 'raft': DATASET_to_RAFT_CKPT_PATHS['sintel']}
        i3d_weights_paths = {
            'rgb': './Feature_Extractor/models/i3d/checkpoints/i3d_rgb.pt',
            'flow': './Feature_Extractor/models/i3d/checkpoints/i3d_flow.pt',
        }
        name2module = {}

        if "flow" in self.streams:
            # Flow extraction module
            if self.flow_type == 'pwc':
                from Feature_Extractor.models.pwc.pwc_src.pwc_net import PWCNet
                flow_xtr_model = PWCNet()
            elif self.flow_type == 'raft':
                flow_xtr_model = RAFT()
            # Preprocess state dict
            state_dict = torch.load(flow_model_paths[self.flow_type], map_location='cpu')
            state_dict = dp_state_to_normal(state_dict)
            flow_xtr_model.load_state_dict(state_dict)
            flow_xtr_model = flow_xtr_model.to(self.device)
            flow_xtr_model.eval()
            name2module['flow_xtr_model'] = flow_xtr_model

        # Feature extraction models (rgb and flow streams)
        i3d_stream_models = {}
        for stream in self.streams:
            i3d_stream_model = I3D(num_classes=self.i3d_classes_num, modality=stream)
            i3d_stream_model.load_state_dict(torch.load(i3d_weights_paths[stream], map_location='cpu'))
            i3d_stream_model = i3d_stream_model.to(self.device)
            i3d_stream_model.eval()
            i3d_stream_models[stream] = i3d_stream_model
        name2module['model'] = i3d_stream_models

        return name2module

    def maybe_show_pred(self, stream_slice: torch.Tensor, model: torch.nn.Module, stack_counter: int) -> None:
        if self.show_pred:
            softmaxes, logits = model(stream_slice, features=False)
            print(f'At stack {stack_counter} ({model.modality} stream)')
            show_predictions_on_dataset(logits, 'kinetics')
