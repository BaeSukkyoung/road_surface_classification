clc, clear all
imageFolder = 'C:\Users\스마트 구조실\Desktop\concrete_image\Concrete\Imageset'

imds = imageDatastore(imageFolder, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);

%imageSize = [224 224 3];
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');
% augimds = augmentedImageDatastore(imageSize,imdsTrain,'DataAugmentation',imageAugmenter);


net = resnet101;

numClasses = numel(categories(imdsTrain.Labels));
lgraph = layerGraph(net);

newFCLayer = fullyConnectedLayer(numClasses,'Name','new_fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10);
lgraph = replaceLayer(lgraph,'fc1000',newFCLayer);

newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',newClassLayer);

trainedNet = resnet101;
inputSize = net.Layers(1).InputSize;

imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandYReflection',true);
     %'RandRotation',[0,180]
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
%augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

options = trainingOptions('sgdm', ...
    'MiniBatchSize',20, ...
    'MaxEpochs',100, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',5, ...
    'Verbose',false, ...
    'Plots','training-progress');


%augmentedImdsTrain = augmentedImageDatastore(imageSize,imdsTrain, ...
 %   'DataAugmentation',imageAugmenter, ...
 %   'OutputSizeMode','randcrop');

trainedNet = trainNetwork(augimdsTrain,lgraph,options);

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
