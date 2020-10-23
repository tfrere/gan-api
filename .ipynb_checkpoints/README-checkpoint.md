# Start serving on floydhub

floyd run --data tfrere/datasets/gan-pre-trained/2:input --mode serve


# API 

- /listPretrainedGans
- /getRandomImages
- /get2dMapFromSeeds
- /getImageInterpolationFromSeeds

## ROUTE /listPretrainedGans

List of pretrained gans

### Params
-

### example
curl -sS 'http://0.0.0.0:5000/listPretrainedGans'


## ROUTE /randomImages

Get a list of random images for a given gan

### Params
- number_of_images
- gan_name

### example
curl -sS 'http://0.0.0.0:5000/getRandomImages?number_of_images=100&gan_name=sneakers' > 'getRandomImages?gan_name=sneakers&number_of_images=100.json'


## ROUTE /get2dMapFromSeeds

Get images data sample for a given base images

### Params
- number_of_images (squared root integer)
- gan_name
- seeds

### example
curl -sS 'http://0.0.0.0:5000/get2dMapFromSeeds?gan_name=chinese&number_of_images=625&seeds=1&seeds=2&seeds=3&seeds=4&seeds=5&seeds=6'


## ROUTE /generateImageInterpolationFromSeed

Get image interpolation from seeds

### Params
- number_of_images
- gan_name
- seeds

### example
curl -sS 'http://0.0.0.0:5000/getImageInterpolationFromSeeds?gan_name=african-masks&number_of_images=10&seeds=1&seeds=2&seeds=3&seeds=4&seeds=5&seeds=6'


