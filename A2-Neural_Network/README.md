Simple Neural Network

Building a simple Neural Network from scratch via a implementation of forward/backward propagation process.

To run:
1. Install and load mnist dataset.
2. Create a model of class NeuralNetwork and fill the constractor as needed.
3. Tweak the hyperparameters.
4. Run the .evaluate function to train the model on five randomized folds.
5. If you feel confidante with the model performances, you may ran it on a test (unseen) data with .predict.

Config:
layer_dims=[784, 10, 10, 10], learning_rate=0.009, epoch=50000, batch_size=16)
Average accuracy: 96.55%
Peak: 97.445%