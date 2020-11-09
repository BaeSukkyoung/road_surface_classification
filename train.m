% reset all variables, command window and close all windows 
% before running the code. 
clc, clear all, close all 

%% Load dataset

% unzip is only required when initial download is completed. 
% unzip('Imageset.zip');

imds = imageDatastore('Imageset','FileExtensions', {'.jpg'},...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
imds.Labels;

%Divide dataset by train dataset & test dataset
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, 'randomized');

%% Load Pretrained Network
net = resnet101;    

net.Layers(1)
inputSize = net.Layers(1).InputSize;

%% Replace Final Layers
if isa(net, 'SeriesNetwork')
   Igraph = layerGraph(net.Layers);
else
   Igraph = layerGraph(net);
end

[learnableLayer,classLayer] = findLayersToReplace(Igraph);
[learnableLayer,classLayer]

numClasses = numel(categories(imdsTrain.Labels));

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cmm.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

Igraph = replaceLayer(Igraph,learnableLayer.Name,newLearnableLayer);

newClassLayer = classificationLayer('Name','new_classoutput');
Igraph = replaceLayer(Igraph,classLayer.Name,newClassLayer);

%%freezeWeights
%layers = Igraph.Layers;
%connections = Igraph.Connections;

%layers(1:10) = freezeWeights(layers(1:10));
%Igraph = createLgraphUsingConnections(layers,connections);

%% Data Augmentation
pixelRange = [-30 30];
scaleRange = [0.9 1.1];
miniBatchSize = 10;
MaxEpochs = 1;
InitialLearningRate = 0.001;

imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXscale',scaleRange, ...
    'RandYScale',scaleRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
    
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',MaxEpochs, ...
    'InitialLearnRate', InitialLearningRate, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');

%% Train network
net = trainNetwork(augimdsTrain,Igraph,options);

%% Validation 

% make predictions for validation dataset
YPred = classify(net,augimdsValidation);

% print validation accuracy 
accuracy = mean(YPred == imdsValidation.Labels)

% show classification examples from the network
idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
   title(string(label))
end
