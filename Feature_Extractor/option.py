import argparse

parser = argparse.ArgumentParser(description='I3D and VGGish Extractor Options')
parser.add_argument('--feature_type', default='i3d', help='feature type to be extracted')
parser.add_argument('--on_extraction', default='save_numpy', help='save the extracted features to output folder')
parser.add_argument('--output_path', default='Feature_Extractor/output-I3D', help='output path')
parser.add_argument('--video_paths', default=None, help='single video path')
parser.add_argument('--file_with_video_paths', default='Feature_Extractor/sample/sample_video_paths.txt', help='A path to a text file with video paths (one path per line). Hint: given a folder ./dataset with .mp4 files one could use: find ./dataset -name "*mp4" > ./video_paths.txt.s')
parser.add_argument('--device', default='cuda:0', help='The device specification. It follows the PyTorch style. Use "cuda:3" for the 4th GPU on the machine or "cpu" for CPU-only.')
parser.add_argument('--show_pred', default=True, help='If true, the script will print the predictions of the model on a down-stream task. It is useful for debugging.')
parser.add_argument('--streams', default='rgb', help='I3D is a two-stream network. By default (null or omitted) both RGB and flow streams are used. To use RGB- or flow-only models use rgb or flow.')
parser.add_argument('--flow_type', default='pwc', help='By default, the flow-features of I3D will be calculated using optical from calculated with PWCNet (originally with TV-L1). Another supported model is raft.')
parser.add_argument('--extraction_fps', default=25, help='If specified (e.g. as 5), the video will be re-encoded to the extraction_fps fps. Leave unspecified or null to skip re-encoding.')
parser.add_argument('--step_size', default=24, help='The number of frames to step before extracting the next features.')
parser.add_argument('--stack_size', default=24, help='The number of frames from which to extract features (or window size).')
parser.add_argument('--tmp_path', default='Feature_Extractor/tmp', help='A path to a folder for storing temporal files (e.g. reencoded videos).')
parser.add_argument('--keep_tmp_files', default=True, help='If true, the reencoded videos will be kept in tmp_path.')
parser.add_argument('--batch_size', default=64, help='If true, the reencoded videos will be kept in tmp_path.')