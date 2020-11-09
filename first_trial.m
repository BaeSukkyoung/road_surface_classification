%Load dataset
unzip('Concrete.zip');
imds = imageDatastore('Concrete');
%Divide dataset by train dataset & test dataset
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, 'randomized');

numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages, 4);
figure
for i = 1:4
    subplot(2, 2, i)
    I = readimage(imdsTrain, idx(i));
    imshow(I)
end
