# Decloud

Decloud enables the training and inference of various neural networks to remove clouds in optical images.

Representative illustrations:

![](doc/images/cap2.jpg)
![](doc/images/cap1.jpg)

*Examples of de-clouded Sentinel-2 images using the single date SAR/Optical U-Net model.*

## Quickstart: Run a pre-trained model
Some pre-trained models are available at this [url](https://nextcloud.inrae.fr/s/DEy4PgR2igSQKKH). 

The easiest way to run a model is to run the timeseries processor such as: 

<pre><code>python production/meraner_timeseries_processor.py
<span style="padding:0 0 0 90px;color:blue">--s2_dir</span>  S2_PREPARE/T31TCJ 
<span style="padding:0 0 0 90px;color:blue">--s1_dir</span>  S1_PREPARE/T31TCJ
<span style="padding:0 0 0 90px;color:blue">--model</span>   merunet_occitanie_pretrained/
<span style="padding:0 0 0 90px;color:blue">--dem</span>     DEM_PREPARE/T31TCJ.tif
<span style="padding:0 0 0 90px;color:blue">--out_dir</span> meraner_timeseries/
<span style="padding:0 0 0 90px;color:grey">--write_intermediate --overwrite</span>
<span style="padding:0 0 0 90px;color:grey">--start</span> 2018-01-01 <span style="color:grey">--end</span> 2018-12-31 
<span style="padding:0 0 0 90px;color:grey">--ulx</span> 306000 <span style="color:grey">--uly</span> 4895000 <span style="color:grey">--lrx</span> 320000 <span style="color:grey">--lry</span> 4888000
</code></pre>
*(mandatory arguments in blue, optional arguments in grey)*

You can find more info on available models and how to use these models [here](doc/pretrained_models.md)



## Advanced usage: Train you own models

1. Prepare the data: convert Sentinel-1 and Sentinel-2 images in the right format (see the [documentation](doc/user_doc.md#Part-A:-data-preparation)).
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


