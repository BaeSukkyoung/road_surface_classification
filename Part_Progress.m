%Load dataset
unzip('Imageset.zip');
imds = imageDatastore('C:\Users\스마트 구조실\Desktop\concrete_image\Concrete\Imageset','LabelSource','foldername','FileExtensions', {'.jpg'})
imds.Labels;

%Divide dataset by train dataset & test dataset
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, 'randomized');

%Load Pretrained Network
net = resnet101;    %=[net = resnet101('Weights', 'imagenet')]

inputSize = net.Layers(1).InputSize;

%Layers 속성의 마지막 요소는 분류 출력 계층, 이 계층의 ClassNames 속성은 신경망이 학습한 클래스의 이름을 포함, 
%총 1,000개 중에서 임의로 10개의 클래스 이름 표시
classNames = net.Layers(end).ClassNames;
numClasses = numel(classNames);
disp(classNames(randperm(numClasses,10)));

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
