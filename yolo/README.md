
# 環境


```
sudo apt-get -qq install -y libsm6 libxext6 && pip install -q -U opencv-python
```


# データ取得


```
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
# wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
# wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_11-May-2012.tar
# tar xf VOCtrainval_06-Nov-2007.tar
# tar xf VOCtest_06-Nov-2007.tar
```

# 重み取得

```
wget wget https://pjreddie.com/media/files/yolo.weights
```



# memo

## Why get not good result by Blood Cell Detection dataset.

https://github.com/experiencor/keras-yolo2/issues/258

```
You use too many warmup batches. WARM_UP_BATCHES of 3 is enough.
```

WARM_UP_BATCHES = 100 is too many.

## The loss is high , but not good result.

https://github.com/experiencor/keras-yolo2/issues/300


```
Don't do average
```

The loss donot do average.


