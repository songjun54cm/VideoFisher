__author__ = 'JunSong<songjun54cm@gmail.com>'
import argparse
from extract_frames import extract_video_frames
from get_cnn_features import get_cnn_features
from get_fisher_vector import get_fisher_vector
from train_valid_svm import train_valid_svm
def main(state):
    print('extracting video frames.')
    extract_video_frames(state['data_folder'], state['frame_number'])

    print('get cnn feature.')
    get_cnn_features(state['data_folder'], state['caffe_folder'], state['caffe_model'], state['caffe_layer'])

    print('start learn fisher vectors')
    fisher_data_folder = get_fisher_vector(state['data_folder'], state['vlfeat_folder'], state['gmm_cluster_n'], state['fisher_norm'], state['pca_number'], state['sampled_number'])

    print('start train svm')
    train_valid_svm(state['data_folder'], state['json_file'], fisher_data_folder, state['svm_c'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_folder', dest='data_folder', type=str, default='/home/vis/songjun/data/activity_net')
    parser.add_argument('--json_file', dest='json_file', type=str, default='activity_net.v1-3.min.json')
    parser.add_argument('--frame_number', dest='frame_number', type=int, default=50)
    parser.add_argument('--caffe_folder', dest='caffe_folder', type=str, default='/home/vis/songjun/tools_3rd/packages/caffe_nocudnn')
    parser.add_argument('--caffe_model', dest='cnn_model', type=str, default='ResNet-101')
    parser.add_argument('--caffe_layer', dest='caffe_layer', type=str, default='res5c')
    parser.add_argument('--vlfeat_folder', dest='vlfeat_folder', type=str, default='/home/vis/songjun/tools_3rd/packages/vlfeat-0.9.20/')
    parser.add_argument('--sampled_number', dest='sampled_number', type=int, default='1000000')
    parser.add_argument('--pca_number', dest='pca_number', type=int, default='32')
    parser.add_argument('--gmm_cluster_n', dest='gmm_cluster_n', type=int, default='128')
    parser.add_argument('--fisher_norm', dest='fisher_norm', type=str, default='')
    parser.add_argument('--svm_c', dest='svm_c', type=float, default=100.0)

    args = parser.parse_args()
    state = vars(args)
    main(state)