
# List pretrained gans
curl -sS 'http://0.0.0.0:5000/listPretrainedGans'

# Get a list of random images for a given gan
curl -sS 'http://0.0.0.0:5000/randomImages?number_of_images=6&gan_name=old_photos'

# Get 625 images data sample for a given base images
curl -sS 'http://0.0.0.0:5000/get2dMapFromSeeds?gan_name=old_photos&seeds=123&seeds=1345&seeds=13453&seeds=134532&seeds=134532&seeds=13452321'