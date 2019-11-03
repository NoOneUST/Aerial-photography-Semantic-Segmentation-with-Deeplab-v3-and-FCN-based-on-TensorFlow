# Aerial photography Semantic Segmentation with Deeplab v3+ based on TensorFlow

## Overview
In this work, we do semantic segmentation based on Google's Deeplab V3+ to classify different areas like the lawn, houses, rivers and roads, etc. on our private aerial images. Our mean IOU is 0.5081. The IOU on each category is 0.214 0.88 0 0.524 0.83 0.686 0.428 respectively.

## Experiment Environment
Parameter | values
----------|---------|
Data unbalance Ratio| 7; 4; 0; 19; 5.5; 3; 16|
number clones| 3
base learning rate |0.0001
learning rate decay factor| 0.1
learning rate decay step| 2000
training number of steps| 30000
momentum |0.9
train batch size |15
weight decay| 0.00004
train crop size| [513, 513]
evaluation crop size| [801, 1201]
atrous rate| 6/12/18
output stride |16
visual output stride| 4

```
Train:

--logtostderr \
--num_clones=3
--training_number_of_steps=100000 \
--train_split="train" \
--model_variant="xception_65" \
--atrous_rates=6 \
--atrous_rates=12 \
--atrous_rates=18 \
--output_stride=16 \
--decoder_output_stride=4 \
--train_crop_size=513 \
--train_crop_size=513 \
--train_batch_size=15 \
--dataset="pascal_voc_seg" \
--train_logdir="~/Tensorflow_Models/models/research/deeplab/Train_Check_Point/" \
--dataset_dir="~/Tensorflow_Models/models/research/deeplab/datasets/pascal_voc_seg/tfrecord/"
--save_interval_secs=60
--save_summaries_secs=30
--base_learning_rate=0.0001
--tf_initial_checkpoint="~/deeplabv3_pascal_trainval/model.ckpt" \

Evaluation:

--logtostderr \
--eval_split="val" \
--model_variant="xception_65" \
--atrous_rates=6 \
--atrous_rates=12 \
--atrous_rates=18 \
--output_stride=16 \
--decoder_output_stride=4 \
--eval_crop_size=801 \
--eval_crop_size=1201 \
--dataset="pascal_voc_seg" \
--checkpoint_dir="~/Tensorflow_Models/models/research/deeplab/Train_Check_Point" \
--eval_logdir="~/Tensorflow_Models/models/research/deeplab/Eval_Check_point" \
--dataset_dir="~/Tensorflow_Models/models/research/deeplab/datasets/pascal_voc_seg/tfrecord/"
--eval_interval_secs=30
--max_number_of_evaluations=1


Visual:

--logtostderr \
--vis_split="val" \
--also_save_raw_predictions = True \
--model_variant="xception_65" \
--atrous_rates=6 \
--atrous_rates=12 \
--atrous_rates=18 \
--output_stride=16 \
--decoder_output_stride=4 \
--vis_crop_size=801 \
--vis_crop_size=1201 \
--dataset="pascal_voc_seg" \
--checkpoint_dir="~/Tensorflow_Models/models/research/deeplab/Train_Check_Point" \
--vis_logdir="~/Tensorflow_Models/models/research/deeplab/Visual_Check_Point" \
--dataset_dir="~/Tensorflow_Models/models/research/deeplab/datasets/pascal_voc_seg/tfrecord/"
--max_number_of_iterations=1


TASK2 DataSet Transform:

--image_folder="~/Tensorflow_Models/models/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VisionTask2/JPEGImages" \
--semantic_segmentation_folder="~/Tensorflow_Models/models/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VisionTask2/SegmentationClassRaw" \
--list_folder="~/Tensorflow_Models/models/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VisionTask2/ImageSets/Segmentation" \
--image_format="png" \
--output_dir="~/Tensorflow_Models/models/research/deeplab/datasets/pascal_voc_seg/tfrecord"


TASK2 Test Transform:

--image_folder="~/Tensorflow_Models/models/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VisionTask2/JPEGImages" \
--semantic_segmentation_folder="~/Tensorflow_Models/models/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VisionTask2/SegmentationClassRaw" \
--list_folder="~/Tensorflow_Models/models/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VisionTask2/ImageSets/Segmentation" \
--image_format="png" \
--output_dir="~/Tensorflow_Models/models/research/deeplab/datasets/pascal_voc_seg/tfrecord"


Original Dataset Transform:
--image_folder="~/Tensorflow_Models/models/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VOC2012/JPEGImages" \
--semantic_segmentation_folder="~/Tensorflow_Models/models/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VOC2012/SegmentationClassRaw" \
--list_folder="~/Tensorflow_Models/models/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VOC2012/ImageSets/Segmentation" \
--image_format="jpg" \
--output_dir="~/Tensorflow_Models/models/research/deeplab/datasets/pascal_voc_seg/tfrecord"


Path:

~/Tensorflow_Models/models/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VOC2012/JPEGImages
~/Tensorflow_Models/models/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VOC2012/SegmentationClassRaw
~/Tensorflow_Models/models/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VOC2012/ImageSets/Segmentation
~/Tensorflow_Models/models/research/deeplab/datasets/pascal_voc_seg/tfrecord

Original 513 513
```
