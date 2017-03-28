__author__ = 'JunSong<songjun54cm@gmail.com>'
import os
import caffe
import numpy as np
import cPickle as pickle
def get_cnn_features(data_folder, caffe_folder, caffe_model, caffe_layer):
    frame_folder = os.path.join(data_folder, 'video_frames')
    feature_folder = os.path.join(data_folder, 'video_feats')
    model_folder = os.path.join(caffe_folder, 'model', caffe_model)

    MODEL_FILE = os.path.join(model_folder, '%s-deploy.prototxt'%caffe_model)
    PRETRAINED = os.path.join(model_folder, '%s-model.caffemodel'%caffe_model)
    MEAN = os.path.join(model_folder, '%s_mean.npy'%caffe_model)

    device_id = 1
    caffe.set_device(device_id)
    caffe.set_mode_gpu()
    net = caffe.Classifier(MODEL_FILE, PRETRAINED, mean=np.load(MEAN).mean(1).mean(1), channel_swap=(2, 1, 0),
                           raw_scale=255, image_dims=(256, 256))


    if not os.path.exists(feature_folder):
        os.makedirs(feature_folder)
    vids = os.listdir(frame_folder)
    count_success = 0
    count_failed = 0
    for vid in vids:
        video_frame_folder = os.path.join(frame_folder, vid)
        feature_file_path = os.path.join(feature_folder, '%s.pkl' % vid)
        # try:
        extract_and_save_feature(net, caffe_layer, frame_folder, feature_file_path)
        print 'extract video %s finish.' % vid
        count_success += 1
        # except:
        #     print 'error while extract %s' % vid
        #     count_failed += 1
        print 'success %d, failed %d' % (count_success, count_failed)

    print 'finish extract frame features success %d videos, failed %d videos' % (count_success, count_failed
                                                                                 )

def extract_and_save_feature(net, caffe_layer, frame_folder, feature_file_path, frame_n=50):
    image_names = sorted(os.listdir(frame_folder))
    use_images = [image_names[int(i)] for i in np.linspace(0, len(image_names)-1, frame_n)]

    temp_img = use_images[0]
    input_image = caffe.io.load_image(os.path.join(frame_folder, temp_img))
    pred = net.predict([input_image], oversample=False)
    # feature with shape 2048x7x7 for res5c layer
    feature = net.blobs[caffe_layer].data[0]

    # format feature to Nx49x2048 for res5c layer
    feat_all = np.zeros((len(use_images), feature.shape[1]*feature.shape[2], feature.shape[0]), dtype=np.float32)

    num = -1
    for img in sorted(use_images):
        num += 1
        if num % 100 == 0:
            print('%d, %s\r'%(num, img)),

        fullName = os.path.join(frame_folder, img)
        input_image = caffe.io.load_image(fullName)

        pred = net.predict([input_image], oversample=False)
        feature = net.blobs[caffe_layer].data[0]
        # print('feature shape: %s' % str(feature.shape))
        feature = feature.reshape(feature.shape[0], feature.shape[1]*feature.shape[2]).transpose()

        feat_all[num,:,:] = feature
    print('%d frames extracted.' % num)

    outp = open(feature_file_path, 'wb')
    pickle.dump(feat_all, outp, protocol=pickle.HIGHEST_PROTOCOL)
    outp.close()