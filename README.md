# Code
### Starting from the paper code:
Starting from the train_cvusa.py, it includes the two VGG16 network:
- VGG.py for the ground images
- VGG_cir.py for the satellite images

After setup the sessions and some other parameters the main functioning is that when training the batch is generated from the class function next_pair_batch() for the training and next_batch_scan() (file Polar_Input_Data_Orien_3.py)for the testing phase. Then the tf.session calls the function VGG_13_conv_v2_cir() (cir_net_fov.py file) where passing the ground and satellite features, they are correlated, shifted and cropped, returning a distance vector used to compute the loss or the accuracy in the testing phase.

To tun the file run the following command for training :
```sh
python train_cvusa_fov.py --train_grd_noise 360 --train_grd_FOV $YOUR_FOV --test_grd_FOV $YOUR_FOV
```
- train_grd_noise 360: means that the FOV cutting is done randomly (to simulate an orientation unknown)
- train_grd_FOV and test_grd_FOV for selecting the FOV of ground images for training and testing

For testing:
```sh
python test_cvusa_fov.py --train_grd_noise 360 --train_grd_FOV $YOUR_FOV --test_grd_FOV $YOUR_FOV
```

### About my code:
The functioning is the same, the difference is that both the VGGs are contained in the file VGG_no_session.py, and I created differents Polar_Input_Data_Orien_3.py files based on the preferred test. 
To simulate the original paper code use Polar_Input_Data_Orien_3.py, VGG_no_session.py and cir_net_fov_mb.py.
The command to run files is the same as before, but usually I change the input parameters manually.
