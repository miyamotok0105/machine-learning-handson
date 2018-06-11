
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


