import numpy as np

from filterpy.kalman import KalmanFilter 

class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4)
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = self.convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = 0
    # self.id = KalmanBoxTracker.count
    # KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(self.convert_bbox_to_z(bbox))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(self.convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return self.convert_x_to_bbox(self.kf.x)

  def convert_bbox_to_z(self, bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

  def convert_x_to_bbox(self, x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
      return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
      return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))

#-------------------------------------------------------#
#---------------------SORT------------------------------#

class Sort(object):
  def __init__(self, max_age = 1, min_hits = 3, iou_threshold = 0.1):
    '''
      输入包含：
      1.连续检测 N次才会被视为命中
      2.丢失M帧会被视为track丢失
      3.前后帧匹配的条件，比如IOU，中心点距离
    '''
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_th = iou_threshold

    self.trackers = []
    self.max_id = 0

  def update(self, dets):

    dets = [det for det in dets if det[-1] >= 0.2]
    ret = []

    # 初始化trackers预测的结果
    trks = np.zeros((len(self.trackers), 5))

    for i, trk in enumerate(trks):
      if len(dets) > 0:  # 检测到目标
        pos = self.trackers[i].predict()[0]
        trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]

        trk_bbox = trk[:4]
        dets_bboxs = np.array(dets)[:, :4]
        match_matrix = self.iou_batch(np.array([trk_bbox]), dets_bboxs)
        best_match_id = np.argmax(match_matrix, axis=-1)
        best_match_iou = match_matrix[0][best_match_id]

        # print(best_match_id)
        # print(best_match_iou)

        # 匹配上的数据
        if best_match_iou > self.iou_th:
          best_match = dets[best_match_id[0]]

          self.trackers[i].update(best_match)
          del dets[best_match_id[0]]  # 将匹配上的det从dets里面删除
        # 没有匹配上的数据

          if self.trackers[i].hit_streak == self.min_hits:
            self.trackers[i].id = self.max_id + 1
            self.max_id = self.max_id + 1

    # 没有匹配到的新的dets
    for i, det in enumerate(dets):
      trk = KalmanBoxTracker(det)
      self.trackers.append(trk)

    # num_trackers = len(self.trackers)
    for j, trk in enumerate(self.trackers):
      d = trk.get_state()[0]

      if (trk.hit_streak >= self.min_hits) and (trk.time_since_update <= self.max_age):
        ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))

      if (trk.time_since_update > self.max_age):
        self.trackers.pop(j)

    if (len(ret) > 0):
      return np.concatenate(ret)
    return np.empty((0, 5))


  def associate_detections_to_trackers(self, detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):
      return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = self.iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
      a = (iou_matrix > iou_threshold).astype(np.int32)
      if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
      else:
        matched_indices = self.linear_assignment(-iou_matrix)
    else:
      matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
      if (d not in matched_indices[:, 0]):
        unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
      if (t not in matched_indices[:, 1]):
        unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
      if (iou_matrix[m[0], m[1]] < iou_threshold):
        unmatched_detections.append(m[0])
        unmatched_trackers.append(m[1])
      else:
        matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
      matches = np.empty((0, 2), dtype=int)
    else:
      matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

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

  def linear_assignment(self, cost_matrix):
    try:
      import lap
      _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
      return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
      from scipy.optimize import linear_sum_assignment
      x, y = linear_sum_assignment(cost_matrix)
      return np.array(list(zip(x, y)))




