# Convolutional SVM vs CNN on adversarial examples
This is a test application to see if Linear SVM on the top of convolutional layers are more robust agaisnt adversarial examples rather than fully connected layer.
Dataset is cifar10 with 50k train, 10k test images and 9933 adversarial images

## Installing

Download pretrained model to save some time :)

You can also use pre created adversarial numpy arrays to save some more time. (adversarial_x_test.npy  and adversarial_y_test.npy)

or in case you may want to create more and more adversarial images -> install foolbox

* [Foolbox](https://github.com/bethgelab/foolbox) - Python toolbox to create adversarial examples that fool neural networks in PyTorch, TensorFlow, Keras...

```
pip install foolbox
```

## Running the tests

* [CNN.ipynb] -  Contains CNN model that is used on creating base model for SVM as well as being substitute model for adversarial attacks. 
* [Adversarial.ipynb] - Creates adversarial examples on test set, the save them as numpy arrays to use in tests.
* [CNN_SVC.ipynb] - Contains Convolution + Linear SVC model training and then evaluatate models and show results. 
* [Main.ipynb] - To debug, show and dig into adversarials.

## Results

Table shows accuracies. CSVM stands for "Convolutional layers + SVM" model.

![testresults](https://user-images.githubusercontent.com/3315340/51072624-df4e9f80-1674-11e9-9cfa-0bee3ebb68b7.png)
