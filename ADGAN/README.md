### Introduction
This is Keras Implementation of 'Anomaly Detection with Generative Adversarial Networks'. 

### Anomaly Detection with Generative Adversarial Networks (ADGAN)
Authors: L. Deecke, R. Vandermeulen, L. Ruff, S. Mandt, and M.Kloft 
Published year: 2018
Summary: Detect anomaly using a pre-trained generator. 
For k steps, compare g(z) and x, and modify theta in g and z based on the loss between g(z) and x. After k steps, if there's a big loss, it means that x is out of distribution from training data. 

Contribution of this paper: 
1. Do not use discriminator, which makes it easy to couple ADGAN with any other GAN-based approach.
2. By seeding from multiple areas in the latent space, this paper account for the non-convexity of the underlying optimization.

### Implementation Result

### Dependency
* Tensorflow 2.0
* Numpy
* Matplotlib

### Datasets
* MNIST
* CIFAR10
* LSUN

### References
* Paper:


### Acknowledgements
This project is sponsored by EpiSys Science and SafeAI Lab, the research team working on uncertainty detection in deep learning. 
