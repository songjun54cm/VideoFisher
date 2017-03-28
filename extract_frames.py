__author__ = 'JunSong<songjun54cm@gmail.com>'
import argparse
import os
import cv2

def extract_video_frames(data_folder, frame_number=50):
    video_folder = os.path.join(data_folder, 'videos')
    frame_folder = os.path.join(data_folder, 'video_frames')
    if not os.path.exists(frame_folder):
        os.makedirs(frame_folder)

    video_list = os.listdir(video_folder)
    for i, video in enumerate(video_list):
        video_name = video.split('.')[0]
        video_path = os.path.join(video_folder, video)
        video_frame_folder = os.path.join(frame_folder, video_name)
        if os.path.exists(video_frame_folder):
            if len(os.listdir(video_frame_folder))>frame_number/2:
                continue
        else:
            os.makedirs(video_frame_folder)
        dump_frames(video_path, video_frame_folder, frame_number)
        print('%d-th video finish. \r' % i)

def dump_frames(vid_path, video_frame_folder, frame_number):
    video = cv2.VideoCapture(vid_path)
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = video_frame_folder

    fcount = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    frame_list = []
    for i in xrange(fcount):
        ret, frame = video.read()
        assert ret
        frame_list.append(frame)
    if len(frame_list) <= frame_number:
        valid_frames = frame_list
    else:
        valid_frames = [frame_list[i] for i in np.linspace(0, len(frame_list), num=frame_number, dtype=int)]
    for i, frame in enumerate(valid_frames):
        cv2.imwrite('{}/{:06d}.jpg'.format(out_full_path, i), frame)
    # print '{} done'.format(vid_name)
