# VideoFisher
video classification with fisher vector

## Guide

to run the code, you can start from the main.py.

some packages you need are "caffe","vlfeat", etc.

this code is written based on the dataset ActivityNet.

**Options**

    --date_folder		the_path_of_your_data_folder
    --json_file		the_path_of_json_data_information_file
    --frame_number		how_many_frames_you_need_for_one_video
    --caffe_folder		the_path_of_caffe
    --caffe_model		the_name_of_caffe_model
    --caffe_layer		the_name_of_caffe_layer
    --vlfeat_folder		the_path_of_vlfeat_folder 
    --sampled_number	the_number_of_points_you_need_to_learn_GMM
    --pca_number		the_dimension_after_pca
    --gmm_cluster_n		the_cluster_number_of_gmm
    --fisher_norm		what_kind_of_fisher_vector_normalize_you_need
    --svm_c			the_value_of_C_in_svm