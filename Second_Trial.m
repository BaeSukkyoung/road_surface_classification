%Load dataset
unzip('Imageset.zip');
imds = imageDatastore('C:\Users\스마트 구조실\Desktop\concrete_image\Concrete\Imageset','LabelSource','foldername','FileExtensions', {'.jpg'})
imds.Labels;

%Divide dataset by train dataset & test dataset
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, 'randomized');

%Load Pretrained Network
net = resnet101;    %=[net = resnet101('Weights', 'imagenet')]

inputSize = net.Layers(1).InputSize;

%Read Image from Dataset
I = imread('P-041-E1_s000202000__0,2048)_.jpg');
figure
imshow(I);

%Resize images according to the network inputsize
I = imresize(I,inputSize(1:2));
figure
imshow(I)

%Classify Dataset
[label,scores] = classify(net,I);
label

%Show percentage of image
figure
imshow(I)
title(string(label) + "," + num2str(100*scores(classNames == label),3) + "%");

%Indicate high rank prediction by histogram
[~,idx] = sort(scores, 'descend');
idx = idx(5:-1:1);
classNamesTop = net.Layers(end).ClassNames(idx);
scoresTop = scores(idx);

figure
barh(scoresTop)
xlim([0 1])
title('Top 5 Predictions')
xlabel('Probability')
yticklabels(classNamesTop)




%Replace Final Layers
if isa(net, '

numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages, 4);
figure
for i = 1:4
    subplot(2, 2, i)
    I = readimage(imdsTrain, idx(i));
    imshow(I)
end

net = resnet101;

inputSize = net.Layers(1).InputSize

%Change the last layer
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(imdsTrain.Labels))

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
    softmaxLayer
    classificationLayer];

%Training the network
pixelRange = [-30, 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandXTranslation', pixelRange, ...
    'RandYTranslation', pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', imageAugmenter);