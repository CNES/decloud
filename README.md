# Decloud

Decloud enables the training of various deep nets to remove clouds in optical images.

Representative illustrations:

![](doc/images/cap2.jpg)
![](doc/images/cap1.jpg)

*Examples of de-clouded images using the single date SAR/Optical U-Net model.*

## Quickstart: Run a pre-trained model
Some pre-trained models are available. You can find more info on how to use them [here](doc/pretrained_models.md)

## Advanced usage: Train you own models

1. Prepare the data: convert Sentinel-1 and Sentinel-2 images in the right format (see the documentation).
![](doc/images/step_1.png)
2. Create some *Acquisition Layouts* (.json files) describing how the images are acquired, ROIs for training and validation sites, and generate some TFRecord files containing the samples.
![](doc/images/step_2.png)
3. Train the network of your choice. The network keys for input/output must match the keys of the previously generated TFRecord files.
![](doc/images/step_3.png)
4. Perform the inference on real world images.
![](doc/images/step_4.png)

More info [here](doc/user_doc.md).

## Contact

You can contact remi cresson (Remi Cresson at INRAE )


