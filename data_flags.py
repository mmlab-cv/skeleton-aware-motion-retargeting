from absl import flags
import os

FLAGS = flags.FLAGS

# DIRECTORIES

flags.DEFINE_string('dataset_path', "", 'Dataset directory.')
flags.DEFINE_string('dataset','CMU','Choose dataset: CMU, MIXAMO, HumanDog')
flags.DEFINE_string('save_path',"",'path to save bvh results')
flags.DEFINE_string('save_path_eval',"",'path to save bvh results')
flags.DEFINE_string('load_checkpoint',"","Checkpoint Path")
# input-output
flags.DEFINE_integer('n_joints', 31, 'Number of target joints. CMU:31, MIXAMO: 21/24, HUMANDOG: 26')
# demo
flags.DEFINE_string('char_input','', 'Name of character for bvh loading')
flags.DEFINE_string('char_target','', 'Name of target character for bvh loading in demo')
flags.DEFINE_string('bvh_name','', 'Name of target character for bvh loading in demo')
# Training Parameters
flags.DEFINE_bool('use_cls',False,'If use class token')
flags.DEFINE_float('lr', 2e-4, 'Learning Rate')
flags.DEFINE_float('b1', 0.5, 'Beta1')
flags.DEFINE_float('b2', 0.999, 'Beta2')
flags.DEFINE_integer('num_workers', int(os.cpu_count() / 2), 'Number of workers.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('n_epochs', 25, 'Number of training epochs.')
flags.DEFINE_integer('masked_joints', 2, 'Number of masked joints: 13,14,6')
flags.DEFINE_string('masking_mode', 'zero', 'Choose masking mode. random, noise, end_effectors')
flags.DEFINE_integer('window_size', 8, 'Size of the window')
flags.DEFINE_float('frame_time', 0.033333, 'Set frame time')

#Mode
flags.DEFINE_string('retargeting','iso','Type of Retargeting')
flags.DEFINE_string('mode','demo','Train Mode')
flags.DEFINE_integer('cuda_device',0,'GPU id')
flags.DEFINE_bool('use_gpu', True, 'Use GPU or not')