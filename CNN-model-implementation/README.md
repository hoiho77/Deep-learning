# Implement CNN models on CIFAR-10 using tensorflow(keras)
cnn model architecture build
- AlexNet

# Environments
- tensorflow
-- tensorflow==2.2.0 , Keras==2.4.3

# Training
To train the model, run 'python train.py'
```
-- model : model you want to train
-- lr    : learning rate
-- epoch : epoch
-- batch : batchsize
```

# Performance Result
performance is based on top1-accuracy
(Training is performed on the 60% of cifar-10 datasets due to the limits of memory)

Model|Epoch|Lr|top1-accuracy|Dataset|
---|---|---|---|---|
AlexNet|50|0.001|0.6791|cifar-10|
-|-|-|-|-|-|
-|-|-|-|-|-|
