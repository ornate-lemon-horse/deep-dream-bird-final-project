# Dreaming of Birds: Exploring Deep Dream Generative Phenomena in Computer Vision

## Michael Shuen

**Link to Slide Deck:** [https://docs.google.com/presentation/d/e/2PACX-1vTfND-ol1-7Wh64Dvsin7vl2rxdnrr7XVyOfFmUAwJftXnefpcfqaHMztYjyWbSl-eA6eNK3ZyhvnLh/pub?start=false&loop=false&delayms=10000](https://docs.google.com/presentation/d/e/2PACX-1vTfND-ol1-7Wh64Dvsin7vl2rxdnrr7XVyOfFmUAwJftXnefpcfqaHMztYjyWbSl-eA6eNK3ZyhvnLh/pub?start=false&loop=false&delayms=10000)

**Link to video:**

## Abstract

In this project, I sought out to explore how the "deep dream phenomena" arose and its properties. In particular, I wanted to find out how to generate interesting dream images and how they would be affected by changes in the underlying model. I trained a model based on the bird classifier from CSE455 and used it to generate deep dream images under various parameters. The work shows that this model is capable of producing convincing images.

## Problem Statement

For this project, I posed the question of how deep dream behavior could be created and how this behavior would change with things like the progression of model training.

## Related Work

The training data that I used was the Bird Project classifer data from the computer vision class, first resized to 128x128. I used the same model as I did then, a pre-trained Inception_v4 network from the `timm` library.

Some inspiration for scoring the model was taken from a TensorFlow blog post on deep dreams: [https://www.tensorflow.org/tutorials/generative/deepdream](https://www.tensorflow.org/tutorials/generative/deepdream)

## Methodology

The "forward" training of the model on the bird dataset was unremarkable: a standard training methodology was used.

To generate the "dream" images, I started from a set of input images, only one of which was used for the outputs shown (more on that later). I used a `DataLoader` to supply these images to the dreaming process. For each iteration, the image was passed through the network and a prediction was made as to which class it belonged to. Then, the network was informed of the "correct" class (the class we wished to emulate), and a gradient was calculated. The model was frozen, so no updates were made to its layers. Instead, the gradient was propagated backwards all the way to the input image, where updates were made. Then, some intermediate processing was done on this generated image, which included clamping the pixel values and performing a small Gaussian blur, as well as a random horizontal flip. This process was then repeated many times.

The loss function that I used for this dreaming process was something that I worked on over the course of this project. Initially, I used the `CrossEntropyLoss` of the predicted and "correct" labels, but I later switched to directly taking the values of the neurons in the final connected layer, which will be discussed below.

## Experiments and Results

I explored how the model performed with and without various levels of training on the bird dataset. The number of neurons in the final layer was different (555 vs. 1000) compared to ImageNet, so the model initally had random weights in this final layer. As expected, the images generated did not resemble the training set:

![image of dream output prior to training on the bird model. There is no bird-like appearance. The output appears like wavy patterns.](0-prelims/new-loss-without-negate-before-training.png)

### Aside on the output format

Due to some issues with the training process, the methodology that eventually worked out results in very tiny changes to the image. The "raw" output from the network is shown on the left side, and the right side shows the output "rescaled" to the 0-255 range.

### Initial challenges

Initially, I faced some challenges with getting this process to work at all. Even after one training epoch, the values of the output images would spiral off towards infinity. I tried some suggestions, like clamping the image to the [0,1] range after every pass, and a small amount of Gaussian blur. However, the actual problem was that the inital learning rate (`lr=0.01`) copied over from model training was too high. After reducing the learning rate several orders of mangnitude, experimentation could really begin. I first trained the Inception_v4 model for 10 epochs on the 128x128 bird dataset, then ran 200 epochs of the dream process.

However, the output produced by this process was relatively unsatisfying.

![dream output after training with old loss function](0-prelims/old-loss-after-train.png)

### New Loss Function

I theorized at this stage that my choice of loss function might be playing a role. The `CrossEntropyLoss` formulation caused the network to try and make the generated image more like a target class, *but also less like all other classes!* This was not what I wanted; the optimal incentive would be to only care about the single class of interest and to not even consider the other classes. Therefore, I changed the loss function to just be the negative of the activation of the neuron in the final layer corresponding to the desired class. This way, gradient descent would actually seek to increase that neuron's activation. Initally, this resulted in a different "feel" to the outputs, but no birds yet:

![output from the model with new loss function after training](0-prelims/new-loss-with-negate-after-training.png)
![another output from the model with new loss function after training](0-prelims/new-loss-with-negate-after-training-with-softmax.png)

After increasing the learning rate significantly, though, there were promising signs:

![output from model after lr increase 1](1-initial-success/1.png)
![output from model after lr increase 2](1-initial-success/2.png)
![output from model after lr increase 3](1-initial-success/3.png)
![output from model after lr increase 4](1-initial-success/4.png)

After this initial success, I attempted to strengthen the images and make them more interesting. 

One approach that didn't seem to provide a benefit was varying the strength of the guassian blur after each pass. I tried varying the sigma in the ranges `(0.1, 1.0)` and `(0.1, 0.5)` (compared to a fixed sigma of 0.5 previously), and the image quality appeared to degrade slightly:

![output from model with variable gaussian blur size](2-varying-gaussian-sigma/1.png)
![output from model with variable gaussian blur size](2-varying-gaussian-sigma/2.png)
![output from model with variable gaussian blur size](2-varying-gaussian-sigma/3.png)

On the other hand, increasing the number of passes through the dreaming process significantly improved image quality, up to 500 epochs.

![output from model with 500 dream steps](3-500-epochs/1.png)
![output from model with 500 dream steps](3-500-epochs/2.png)
![output from model with 500 dream steps](3-500-epochs/3.png)
![output from model with 500 dream steps](3-500-epochs/4.png)
