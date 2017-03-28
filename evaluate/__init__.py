import numpy as np
import json

def generate_prediction_file(video_ids, predictions, label_id2name, out_file_path,
                             topk=3, version='VERSION 1.3', external_info=''):
    prediction_info = {
        'version': version,
        'results': {},
        'external_data': {
            'used': False,
            'details': ''
        }
    }
    if external_info:
        prediction_info['external_data'] = {
            'used': True,
            'details': external_info
        }

    count = 0
    for vid, pred in zip(video_ids, predictions):
        count += 1
        # if count%100 == 0:
            # print 'count: %d, vid: %s\r' % (count, vid)
        topidx = np.argsort(pred)[::-1][:topk]
        res = list()
        for idx in topidx:
            # print('idx: ', idx)
            res.append(
                {
                    'label': label_id2name[idx],
                    'score': float(pred[idx])
                }
            )
        # print res
        # print('vid: %s, res: %s' % (vid, json.dumps(res)))
        prediction_info['results'][vid] = res
    with open(out_file_path, 'wb') as f:
        json.dump(prediction_info, f, indent=2)
    return out_file_path

def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap

def get_blocked_videos(api='http://ec2-52-11-11-89.us-west-2.compute.amazonaws.com/challenge16/api.py'):
    import json
    import urllib2
    api_url = '{}?action=get_blocked'.format(api)
    req = urllib2.Request(api_url)
    response = urllib2.urlopen(req)
    return json.loads(response.read())