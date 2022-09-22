# Grain_classify
Different Deep Neural networks are analysed and trained to present a Grain size classification system of superalloy Images. 
Images were obtained at previous study by Ph.D. Pedro Paramo Kañetas. Four different Neural Networks were selected: AlexNet, MobileNet V2, Resnet50 and DenseNet 201. Classification performance was measured in accuracy awith the highest of 0.824 achieved by Densenet 201. The three classification levels are Fine, Medium  and Big. Table 1 shows the  total set of images, 177, divided into the three different clases. Most of the images of size 1024x1280.
Number of images
Desciption |Total images | 
--- | --- | 
Fine  | 87 | 
Medium| 47 | 
Big | 43 | 

RGM colour images
Size 1024X1280 

Example of Fine grain images

![Samples_Finos](https://user-images.githubusercontent.com/44585823/191774552-80d43439-e011-48ab-8247-714abd7854c3.png)

Example of Medium grain images

![Samples_Medianos](https://user-images.githubusercontent.com/44585823/191774576-6e88022b-82c5-415a-875c-32458491104c.png)


Example of Big grain images


![Samples_Grandes](https://user-images.githubusercontent.com/44585823/191774593-707191be-cd5c-469d-8017-3998b5918de2.png)


Networks selected were: AlexNet, Mobile V2, Resnet 50 and Densenet 201


![networks_examples](https://user-images.githubusercontent.com/44585823/171680445-4b4e076c-6f8f-4df0-9a3f-3a589c283d70.png)



The data was splitted into image patches od 224x224, and after data augmentation of 90, 180 degrees rotation, fip up/down  and flip left/right
Two new sets of training and validation images were obtained:

Training set
Desciption |Total images | 
--- | --- | 
Fine  | 2432 | 
Medium| 1776 | 
Big | 1872 | 

Validation set
Desciption |Total images | 
--- | --- | 
Fine  | 320 | 
Medium| 288 | 
Big | 384 | 


Different networks were trainned and although a high accuracy was obtained after training, Validation was performed with a selected group of completely new images not seen during training.


```
filenameTE = fullfile('D:\300123\Documentos\Proyecto_GrainSize\PreparadosColor\');
imds = imageDatastore(filenameTE, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.8,'randomized');
numClasses = numel(categories(imdsTrain.Labels));





% Aqui se abre la red a usar
%net=googlenet;
net=alexnet;
%net=resnet50;
%net=mobilenetv2
%net=densenet201;

inputSize = net.Layers(1).InputSize;
if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 
[learnableLayer,classLayer] = findLayersToReplace(lgraph);
numClasses = numel(categories(imdsTrain.Labels));

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);

pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

miniBatchSize = 10;
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);

options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');


[net, info]= trainNetwork(augimdsTrain,lgraph,options);
```


![Training_process](https://user-images.githubusercontent.com/44585823/191779577-42eccf52-eabb-45ae-bff0-d9b9303425c3.png)

Performance of classificationn was evaluated by means of accuracy. Accuracy of segmentation is obtained by the following expression:

   <img src="https://latex.codecogs.com/svg.image?Accuracy=\frac{TP&plus;TN}{TP&plus;FP&plus;TN&plus;FN}" title="Accuracy=\frac{TP+TN}{TP+FP+TN+FN}" />
   
Network |Accuracy | 
--- | --- | 
MobileNetV2  | 0.8196 | 
ResNet 50| 0.7863 | 
Densenet 201| 0.824 | 
Alex net| 0.7843 | 


The best result was obtained with Densenet 201. Its corresponding Confusion matrix is as follow


![ConfusionGood_Densenet](https://user-images.githubusercontent.com/44585823/191781919-58a03441-d9d8-41ea-a780-4fbbf72be71a.jpg)
