%Load dataset
unzip('Imageset.zip');
imds = imageDatastore('C:\Users\스마트 구조실\Desktop\concrete_image\Imageset','FileExtensions', {'.jpg'},...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames')
imds.Labels;

%Divide dataset by train dataset & test dataset
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, 'randomized');

%Load Pretrained Network
net = resnet101;    %=[net = resnet101('Weights', 'imagenet')]

net.Layers(1)
inputSize = net.Layers(1).InputSize;

%Replace Final Layers
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

%Data Augmentation
pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXscale',scaleRange, ...
    'RandYScale',scaleRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
    
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

miniBatchSize = 10;
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',10, ...
    'InitialLearnRate', 0.0001, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(augimdsTrain,Igraph,options);

YPred = classify(trainedNet,augimdsValidation);
accuracy = mean(YPred == imdsValidation.Labels)
idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:8
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
   title(string(label))
end

%Indicate high rank prediction by histogram
[~,idx] = sort(zscore, 'descend');
idx = idx(5:-1:1);
classNamesTop = net.Layers(end).ClassNames(idx);
scoresTop = scores(idx);

figure
barh(scoresTop)
xlim([0 1])
title('Top 5 Predictions')
xlabel('Probability')
yticklabels(classNamesTop)