# What is Hidden in a Randomly Weighted Neural Network?

This is a simple implementation in numpy of [this paper](https://arxiv.org/abs/1911.13299). Not using any framework because i couldn't figure out how to manually control backward pass in TF2/Keras.

To use, might want to replace the data i loaded with your own because its too big to be put on this repo.

Expected input shape :
(batch_size, num_features) for inputs
(batch_size) for labels

TODO : 
- Migrate to any framework
- Training on the ScoreFCNN is super slow. Need to work out on better optimizations.