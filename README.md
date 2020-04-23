The repository contains the code for training and testing for fashion-MNIST dataset. Purpose of each file is as follows:

The `model_training.py` contains the main code for training the models. Different parameters for model training and paths need to be set in the `config.py` which are then used in the `model_training.py` file. Different model architectures and utility functions can be found in `model.py` and `utils.py` respectively. 

Once the training is performed, `model_evaluation.py` is used to test the trained model on unseen images. To try out the code, a trained model weight file is given in `VGG_like_CNN_1_model/best_model.pt`. To see the performance of this model, the paths have already been set to simply run `model_evaluation.py`. If the code runs successfully, we should get the confusion matrix and model accuracy and loss on the test dataset.  
