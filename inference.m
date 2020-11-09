%Load dataset
    % unzip('Imageset.zip'); % code for unzip is not necessary 
    % Recommend not to include root directory in your code. 
imds = imageDatastore('Imageset', 'IncludeSubfolders', true , ...
    'LabelSource','foldername','FileExtensions', {'.jpg'});

%Divide dataset by train dataset & test dataset
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, 'randomized');

%Load Pretrained Network
net = resnet101;    

inputSize = net.Layers(1).InputSize;

%Read Image from Dataset
    % recommend random choose from the image set 
img_list = dir(fullfile('Imageset', string(imds.Labels(1)), '*.jpg')); 
I = imread(fullfile(img_list(1).folder, img_list(1).name));
figure;
imshow(I);

%Resize images according to the network inputsize
I = imresize(I,inputSize(1:2));
figure; 
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




