import numpy as np

class IouTracker(object):
    '''
        在线IouTracker
    '''
    def __init__(self, sigma_l = 0, sigma_h = 0.6, sigma_iou = 0.1, t_min = 3, disappear_time = 5):
        super(IouTracker, self).__init__()
        self.sigma_l = sigma_l
        self.sigma_h = sigma_h
        self.sigma_iou = sigma_iou
        self.t_min = t_min
        self.max_id = 0
        self.disappear_time = disappear_time

        self.tracks_observe = []

    def update(self, dets):
        '''
        需要给每一个dets加上编号,

        :param dets: 当前帧的检测结果
        :return: 给当前帧的检测结果编号
        '''

        self.tracks_finished = [] #返回的轨迹
        self.tracks_matched = [] #当前帧新出现的生命周期以下待考察的轨迹
        self.tracks_unmatched = []

        dets = [det for det in dets if det[-1] >= self.sigma_l]

        for track in self.tracks_observe:
            if len(dets) > 0: # 检测到目标
                # 找到与dets里与当前track_iou最大的目标
                track_bbox = track['bbox']
                dets_bboxs = np.array(dets)[:, :4]
                match_matrix = self.iou_batch(np.array([track_bbox]), dets_bboxs)
                best_match_id = np.argmax(match_matrix, axis=-1)
                best_match_iou = match_matrix[0][best_match_id]

                if best_match_iou > self.sigma_iou:
                    best_match = dets[best_match_id[0]]

                    # 当前帧的track匹配
                    track['bbox'] = best_match[:4] # 更新当前track的bbox
                    track['score'] = best_match[-1] # 更新当前score
                    track['life'] += 1 # 匹配上，则检测长度+1
                    track['disappear'] = 0

                    del dets[best_match_id[0]]  # 将匹配上的det从dets里面删除
                    self.tracks_matched.append(track)

                    # 这里主要用来维护目标的编号信息，根据目标的生命周期分配
                    if track['life'] == self.t_min:
                        # 大于t_min的给与分配id, 同时更新当前max_id
                        track['id'] = self.max_id + 1
                        self.max_id += 1
                        self.tracks_finished.append(track)

                    elif track['life'] > self.t_min:
                        # life > t_min，代表已经分配了id, track_id不变
                        self.tracks_finished.append(track)

                # 对当前轨迹中没有匹配上的数据做处理，
                # 使用KCF对当前轨迹做延长预测，此前的生命周期越长，则后续弥补的帧数越长
                else:
                    track['disappear'] += 1
                    if track['disappear'] < self.disappear_time:
                        self.tracks_unmatched.append(track)

        # id为0，则表示未分配id
        new_tracks = [{'bbox': det[:4], 'score': det[-1], 'life': 1, 'id':0, 'disappear': 0} for i, det in enumerate(dets)]

        self.tracks_observe = self.tracks_matched + new_tracks + self.tracks_unmatched

        return self.tracks_finished

    def iou_batch(self, bb_test, bb_gt):
        """
        From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
        """
        bb_gt = np.expand_dims(bb_gt, 0)
        bb_test = np.expand_dims(bb_test, 1)

        xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
        yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
        xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
        yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
                  + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
        return (o)