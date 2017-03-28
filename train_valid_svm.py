__author__ = 'JunSong<songjun54cm@gmail.com>'
import numpy as np
import scipy.io as sio
import os
import cPickle as pickle
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV

def train_valid_svm(data_folder, json_file, fisher_data_folder, svm_c):
    video_ids, video_feats = get_id_feats(data_folder, fisher_data_folder)
    data_info_path = os.path.join(data_folder, 'data_fisher_info.pkl')
    if not os.path.exists(data_info_path):
        print('parsing file %s' % json_file)
        data_info = parse_data_json(os.path.join(data_folder, json_file))
        pickle.dump(data_info, open(data_info_path, 'wb'))
        print('save data info to %s' % data_info_path)
    else:
        print('loading data info from %s.' % data_info_path)
        data_info = pickle.load(open(data_info_path, 'rb'))
    data_version = data_info['version']
    video_db = data_info['video_db']
    train_data = {'ids': [], 'x':[], 'labels':[]}
    valid_data = {'ids': [], 'x':[], 'labels':[]}
    test_data = {'ids': [], 'x': [], 'labels': []}
    for id,fe in zip(video_ids, video_feats):
        vinfo = video_db.get(id, None)
        if vinfo is not None:
            if vinfo['subset'] == 'training':
                train_data['ids'].append(id)
                train_data['x'].append(fe)
                train_data['labels'].append(vinfo['label'])
            elif vinfo['subset'] == 'validation':
                valid_data['ids'].append(id)
                valid_data['x'].append(fe)
                valid_data['labels'].append(vinfo['label'])
            else:
                test_data['ids'].append(id)
                test_data['x'].append(fe)
    train_data['x'] = np.array(train_data['x'])
    print('train data with shape : %s' % str(train_data['x'].shape))
    valid_data['x'] = np.array(valid_data['x'])
    print('valid data with shape : %s' % str(valid_data['x'].shape))
    test_data['x'] = np.array(test_data['x'])
    print('test data with shape: %s' % str(test_data['x'].shape))

    lsvm = svm.LinearSVC(verbose=True, C=100.0)
    clf = CalibratedClassifierCV(lsvm)
    clf.fit(train_data['x'], train_data['labels'])
    classes = clf.classes_
    model_save_path = os.path.join(fisher_data_folder, 'model_svm.pkl')
    pickle.dump(clf, model_save_path)

    valid_prediction = clf.predict_proba(valid_data['x'])

    from evaluate import generate_prediction_file
    prediction_file_path = os.path.join(fisher_data_folder, 'svm_valid_prediction.txt')
    generate_prediction_file(valid_data['ids'], valid_prediction, classes, prediction_file_path, topk=3, version=data_version)

    from evaluate.eval_classification import ANETclassification
    anet_vlassification = ANETclassification(os.path.join(data_folder, json_file),
                                             prediction_file_path,
                                             subset='validation',
                                             verbose=True,
                                             check_status=False)
    anet_vlassification.evaluate()

def get_id_feats(data_folder, fisher_data_folder):
    print('getting ids and feats.')
    vids = pickle.load(os.path.join(data_folder, 'fisher_mat', 'video_ids.pkl'))
    fisher_vecs = sio.loadmat(os.path.join(fisher_data_folder, 'all_video_feature.mat'))['all_video_feat']
    print('load fisher vectors with shape %s' % str(fisher_vecs.shape))
    return vids, fisher_vecs

def parse_data_json(json_path):
    print('parsing json data.')
    import json
    data_info = dict()
    with open(json_path, 'rb') as f:
        act_data = json.load(f)
    data_info['version'] = act_data['version']
    video_db = dict()
    database = act_data['database']
    # get classes names and ids
    label_names = list()
    for vid,vinfo in database.iteritems():
        video_db[vid] = {'subset': vinfo['subset'], 'label': 'none'}
        annos = vinfo['annotations']
        if len(annos)<1:
            continue
        else:
            vlabel_name = annos[0]['label']
            video_db[vid]['label'] = vlabel_name
        label_names.append(vlabel_name)

    label_names = sorted(list(set(label_names)))
    print('number of labels: %d' % len(label_names))
    label_id2name = label_names
    label_name2id = dict(zip(label_names, range(len(label_names))))
    data_info['label_id2name'] = label_id2name
    data_info['label_name2id'] = label_name2id
    data_info['nb_classes'] = len(label_id2name)
    data_info['video_db'] = video_db
    return data_info