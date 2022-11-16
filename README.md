## IOU_Trackers & SORT algorithm

### 主要工作

#### 1.针对IOU_tracker的检测框丢失重新分配id的情况，为每个tracker设置了一定的生命周期，在没有检测框匹配到的时候，tracker仍然会存在，直到连续disappear_time帧没有检测到，就会删除该tracker



#### 2.针对SORT算法初始化时就会分配id的问题，改为tracker的与bbox关联次数达到min_hits时才会给tracker分配id



### demo

#### IOU_tracker

```
	python3 iou_track_demo.py
```

#### SORT
```
	python3 sort_demo.py
```