# Clothes-RecSys
Clothes RecSys with VAE inside

## Project description
Dataset base on SSENSE website also known as FOTOS dataset was used. SSENSE.com is a
popular fashion website, where user can create and upload outfit data. Dataset consists of 11,000
well-matched outfit composed by 20,384 fashion items. For now I will only use images that are
related to looks consisting of tops-bottoms and bottoms-shoes. So, there are about 8500 outfits.

## Model architecture
Input images had shape 680x680 and were resized to 128x128, latent space has 64 dimensions.
Encoder has 5 convolution layers each with batch normalization and LeakyReLU activation function.

![model](https://user-images.githubusercontent.com/31104632/180863146-57fc959c-78e9-4f74-ab69-586c4215320a.png)

## Recommendations example
![2022-07-24_01-20-58](https://user-images.githubusercontent.com/31104632/180863346-db298680-5f0f-4b6c-8937-aaeede6b3f04.png)
