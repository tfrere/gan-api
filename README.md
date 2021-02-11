# Start serving on floydhub

# floyd run --data tfrere/datasets/gan-pre-trained/2:input --mode serve

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
curl -sS 'http://0.0.0.0:5000/getRandomImages?number_of_images=100&gan_name=african-masks' > 'getRandomImages?gan_name=new-new-african-masks&number_of_images=100.json'

curl -sS 'http://0.0.0.0:5000/getRandomImages?number_of_images=100&gan_name=meal' > 'getRandomImages?gan_name=cocktails&number_of_images=100.json'


## ROUTE /get2dMapFromSeeds

Get images data sample for a given base images

### Params
- number_of_images (squared root integer)
- gan_name
- seeds

### example
curl -sS 'http://0.0.0.0:5000/get2dMapFromSeeds?gan_name=chinese&number_of_images=625&seeds=1&seeds=2&seeds=3&seeds=4&seeds=5&seeds=6'

curl -sS 'http://0.0.0.0:5000/get2dMapFromSeeds?gan_name=meal&number_of_images=625&seeds=1&seeds=2&seeds=3&seeds=4&seeds=5&seeds=6' > 'get2dMapFromSeeds?gan_name=cocktails&number_of_images=625&seeds=1&seeds=2&seeds=3&seeds=4&seeds=5&seeds=6.json'


curl -sS 'http://0.0.0.0:5000/get2dMapFromSeeds?gan_name=handbag&number_of_images=625&seeds=1&seeds=2&seeds=3&seeds=4&seeds=5&seeds=6' > 'get2dMapFromSeeds?gan_name=handbag&number_of_images=625&seeds=1&seeds=2&seeds=3&seeds=4&seeds=5&seeds=6.json'


curl -sS 'http://0.0.0.0:5000/get2dMapFromSeeds?gan_name=branly-african-masks&number_of_images=625&seeds=7071&seeds=7073&seeds=6973&seeds=6911' > 'get2dMapFromSeeds?gan_name=branly-african-masks&number_of_images=625&seeds=7071&seeds=7073&seeds=6973&seeds=6911.json'

curl -sS 'http://0.0.0.0:5000/get2dMapFromSeeds?gan_name=branly-african-masks&number_of_images=625&seeds=7071&seeds=7073&seeds=6973&seeds=6911' > 'get2dMapFromSeeds?gan_name=branly-african-masks&number_of_images=625&seeds=0022&seeds=0027&seeds=0070&seeds=0009.json'




## ROUTE /generateImageInterpolationFromSeed

Get image interpolation from seeds

### Params
- number_of_images
- gan_name
- seeds

### example
curl -sS 'http://0.0.0.0:5000/getImageInterpolationFromSeeds?gan_name=african-masks&number_of_images=10&seeds=1&seeds=2&seeds=3&seeds=4&seeds=5&seeds=6'


curl -sS 'http://0.0.0.0:5000/getImageInterpolationFromSeeds?gan_name=handbag&number_of_images=625&seeds=1&seeds=2&seeds=3&seeds=4&seeds=5&seeds=6' > 'getImageInterpolationFromSeeds?gan_name=handbag&number_of_images=625&seeds=1&seeds=2&seeds=3&seeds=4&seeds=5&seeds=6.json'


curl -sS 'http://0.0.0.0:5000/getImageInterpolationFromSeeds?gan_name=branly-african-masks&number_of_images=625&seeds=1&seeds=2&seeds=3&seeds=4&seeds=5&seeds=6'



curl -sS 'http://0.0.0.0:5000/get2dMapFromSeeds?gan_name=branly-african-masks&number_of_images=625&seeds=6923&seeds=6959&seeds=6962&seeds=6973&seeds=6978&seeds=7015&seeds=7024&seeds=7064&seeds=7095&seeds=7108&seeds=7111&seeds=7177&seeds=7215&seeds=7246&seeds=7281&seeds=7317&seeds=7454&seeds=7465' > 'toto.json'
      