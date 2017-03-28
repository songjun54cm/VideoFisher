__author__ = 'JunSong<songjun54cm@gmail.com>'
import os
from glob import glob
import cPickle as pickle
import numpy as np
import scipy.io as sio
from tqdm import tqdm


def get_fisher_vector(data_folder, vlfeat_folder, gmm_cluster_n, fisher_norm, pca_number, sampled_number):
    feature_file_path = convert_pkl_to_mat(data_folder)

    m_param_file = 'fisher_params.m'
    fisher_folder = os.path.join(data_folder, 'fisher_mat', 'samp%d_pca%d_gmm%d_fnorm%s')
    if not os.path.exists(fisher_folder):
        os.makedirs(fisher_folder)
    with open(m_param_file, 'wb') as f:
        f.write('data_folder = \"%s\";\n' % data_folder)
        f.write('all_feat_file_path = \"%s\";\n' % feature_file_path)
        f.write('vlfeat_dir = \"%s\";' % vlfeat_folder)
        f.write('gmm_cluster_n = %d;\n' % gmm_cluster_n)
        f.write('fisher_norm = \"%s\";\n' % fisher_norm)
        f.write('pca_number = %d;' % pca_number)
        f.write('sampled_number = %d;' % sampled_number)
        f.write('save_folder = \"%s\";' % fisher_folder)
        print('write parameters into %s' % m_param_file)
    print('running matlab')
    os.system("matlab -nojvm -nodisplay -nosplash -nodesktop get_fisher_vector.m")
    return fisher_folder

def convert_pkl_to_mat(data_folder):
    print('converting pkl to mat')
    feature_folder = os.path.join(data_folder, 'video_feats')
    pkl_list = glob(os.path.join(feature_folder, '*.pkl'))
    video_id_list = list()
    video_feat_list = list()
    for pklf in tqdm(pkl_list):
        video_id = pklf.split('/')[-1].split('.')[0]
        video_id_list.append(video_id)
        # video feat with shape nX49x2048 for res5c layer
        video_feat = pickle.load(open(pklf, 'rb'))
        video_feat_list.append(video_feat)
    all_video_feat = np.array(video_feat_list)
    print('all video feature shape: %s' % (str(all_video_feat.shape)))
    mat_folder = os.path.join(data_folder, 'fisher_mat')
    if not os.path.exists(mat_folder):
        os.makedirs(mat_folder)

    video_id_file = os.path.join(mat_folder, 'video_ids.pkl')
    pickle.dump(video_id_list, open(video_id_file, 'wb'))
    print('save video id list into %s'%video_id_file)

    all_video_feat_file = os.path.join(mat_folder, 'all_video_feature.mat')
    sio.savemat(all_video_feat_file, {'all_video_feat': all_video_feat})
    print('save all video feature into %s'%all_video_feat_file)
    return all_video_feat_file

