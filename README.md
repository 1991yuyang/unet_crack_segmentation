# unet_crack_segmentation
unet crack segmentation
## Using Unet to segement the road cracks
* ### configure
> First, you need to configure the training parameters. The configuration file is train_conf.json. Some parameters are explained below.
  
>> valid_good_perform_times: Used to specify the conditions for saving the model, that is, when the model performs well on the validation set for 5 consecutive times, the model is saved.  
  
>> positive_weight: Specify the weight of positive samples(pixel) during training.
  
>> negative_weight: Specify the weight of negative samples(pixel) during training.  
  
>> "is_attention": Specify whether to use the attention mechanism, if set to "True", otherwise "False".  
  
* ### training
> run train.py to train the model.It should be noted that model training consumes more memory. I use GPU 1050Ti (4G memory) for model training. I specify batch_size as 1.  
  
* ### testing
> run predict.py to test one image, The following points need attention.
>> * The structure of the model needs to be consistent with the model structure during training.
>> * Need to specify one image path.

* ### predict result
![image](https://github.com/1991yuyang/unet_crack_segmentation/blob/master/dataset/image_test/320.jpg)
![mask](https://github.com/1991yuyang/unet_crack_segmentation/blob/master/predict_result/320.jpg)

