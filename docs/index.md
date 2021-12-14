# Dreaming of Birds: Exploring Deep Dream Generative Phenomena in Computer Vision

## Michael Shuen

**Link to Slide Deck:** [https://docs.google.com/presentation/d/e/2PACX-1vTfND-ol1-7Wh64Dvsin7vl2rxdnrr7XVyOfFmUAwJftXnefpcfqaHMztYjyWbSl-eA6eNK3ZyhvnLh/pub?start=false&loop=false&delayms=10000](https://docs.google.com/presentation/d/e/2PACX-1vTfND-ol1-7Wh64Dvsin7vl2rxdnrr7XVyOfFmUAwJftXnefpcfqaHMztYjyWbSl-eA6eNK3ZyhvnLh/pub?start=false&loop=false&delayms=10000)

**Link to video:**

## Abstract

In this project, I sought out to explore how the "deep dream phenomena" arose and its properties. In particular, I wanted to find out how to generate interesting dream images and how they would be affected by changes in the underlying model. I trained a model based on the bird classifier from CSE455 and used it to generate deep dream images under various parameters. The work shows that this model is capable of producing convincing images.

## Problem Statement

For this project, I posed the question of how deep dream behavior could be created and how this behavior would change with things like the progression of model training.

## Related Work

The training data that I used was the Bird Project classifer data from the computer vision class. I used the same model as I did then, a pre-trained Inception_v4 network from the `timm` library.

Some inspiration for scoring the model was taken from a TensorFlow blog post on deep dreams: [https://www.tensorflow.org/tutorials/generative/deepdream](https://www.tensorflow.org/tutorials/generative/deepdream)

## Methodology

The "forward" training of the model on the bird dataset was unremarkable: a standard training methodology was used.

To generate the "dream" images, I started from a set of input images, only one of which was used for the outputs shown (more on that later). I used a `DataLoader` to supply these images to the dreaming process. For each iteration, the image was passed through the network and a prediction was made as to which class it belonged to. Then, the network was informed of the "correct" class (the class we wished to emulate), and a gradient was calculated. The model was frozen, so no updates were made to its layers. Instead, the gradient was propagated backwards all the way to the input image, where updates were made. Then, some intermediate processing was done on this generated image, which included clamping the pixel values and performing a small Gaussian blur, as well as a random horizontal flip. This process was then repeated many times.

## Evaluation and Results

