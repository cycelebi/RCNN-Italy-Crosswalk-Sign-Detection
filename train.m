% Mehmet Celebi
% Matricola: 0323916
% University of Rome "Tor Vergata"
% Crosswalk sign detection with RCNN
% Measurement Systems for Mechatronics
% Professor Arianna Mencattini

% R-CNN is Region Based Convolution Neural Network which works with proposed
% region selection to train and detect various objects.

%%
data = load('2cwSignsGroundTruth.mat');
trainingData = objectDetectorTrainingData(data.gTruth);

% loads .mat file '2cwsignsGroundTruth.mat' 
% which contains the ground truth information about traffic signs. 
% The loaded data is stored in a variable named 'gTruth'.
% Ground Truth obtained by MATLAB Image Labeler. 
% Since RCNN is a Region based, each region containing True Positive
% labeled and other parts remained as True Negative

% For Training DITS - the Data set of Italian Traffic Signs used
% http://www.diag.uniroma1.it/~bloisi/ds/dits.html

%%

% Define the layers for the object detector.
layers = [
    imageInputLayer([64 64 3])
    % This code creates an input layer 
    % with a size of 64x64 pixels and 3 color channels (RGB).

    convolution2dLayer(2,16,'Padding','same')
    % Create a 2D convolutional layer. 
    % 2, specifies the size of the filters as 2x2 pixels. 
    % 30, specifies the number of filters to apply to the input image.
    % Increasing gives more accuracy in complex shapes 
    % 'Padding','same' input image should be padded with 
    % zeros so that the output feature map has the same size 
    % as the input. 

    batchNormalizationLayer
    % maintains the mean near 0 and the 
    % standard deviation near 1. 
    % stabilize the training process 
    % and improve the performance. 

    reluLayer
    % any negative values in the input are set to zero, 
    % positive values same. 
    % This non-linearity helps to introduce non-linearity 
    % and increase the model's ability to learn complex patterns 
    % in the input data. It is a widely used activation function due 
    % to its simplicity and effectiveness in deep learning model

    maxPooling2dLayer(2,'Stride',2)
    % creates a 2D max pooling layer for the neural network. 
    % The first argument, 2, specifies the size of the pooling 
    % window as 2x2 pixels. The second argument, 'Stride',2, 
    % specifies the stride of the pooling operation. 
    % Made to reduce spatial size of NN

    fullyConnectedLayer(2)
    % The argument 2 specifies the number of 
    % neurons in the layer. In this project this means Background + Classes
    % which is only "crosswalk" in ground truth
    
    softmaxLayer
    % Applies the softmax function to the input. The softmax function is a 
    % mathematical function that converts a vector of real numbers into a 
    % probability distribution.
    
    classificationLayer
    % the neural network is trained to detect whether an input 
    % image contains a crosswalk sign or not, so the classificationLayer 
    % has two classes: "crosswalk sign" and "not crosswalk sign".
    ];


%%
% Set the options for the object detector.

% sgdm is an optimization algorithm used for training deep learning models, 
% specifically for stochastic gradient descent with momentum. 
% The algorithm is an extension of the standard momentum method, 
% which helps accelerate the convergence of the optimization process by 
% adding a fraction of the previous update to the current update.

% 'MiniBatchSize' specifies the number of images to use in each mini-batch 
% during training. A larger batch size can speed up training, but may 
% require more memory. In this case, a batch size of 128 was used.

% 'InitialLearnRate' sets the initial learning rate 
% for the optimizer. The learning rate controls how quickly 
% the optimizer adjusts the weights of the neural network during training.
% A higher learning rate can cause the optimizer to converge faster, 
% but may result in unstable training. In this case, 
% an initial learning rate of 0.001 was used.

% 'MaxEpochs' sets the maximum number of epochs to train the network. 
% An epoch is a complete pass through the training data. 
% Training may stop earlier if the validation loss stops improving.
% In this case, training was stopped after 50 epochs.

% 'Shuffle' specifies how to shuffle the training data before each epoch. 
% 'every-epoch' shuffles the data at the beginning of each epoch. 
% Shuffling the data can help prevent overfitting and improve training performance.

% 'Verbose' specifies whether to display training progress information 
% in the command window during training. Setting this to true displays 
% the progress.

% 'Plots' specifies whether to display training plots during training.
% Setting this to 'training-progress' displays a plot 
% of the training over time.

%%
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 128, ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 30, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress');
% Train the object detector.
detector = trainRCNNObjectDetector(trainingData, layers, options);

% Validation and performance analysis couldnt perform since research about
% RCNN showed that this algorithm does not support VerificationData
% arguments in options and training.

%%
% Saves detector result to use in test.m
save('1630128cwSignsDetector.mat', 'detector');