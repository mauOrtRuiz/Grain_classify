# Grain_classify
A Neural network method for Grain Classification  of superalloy Images. 
Images were obtained by Ph.D. Pedro ParamoK.  with following characteristics:

Fine
Medium
Big


After a first inspection of data some basic statistics of the images set is obtained. 
```
clear

filenameTE = fullfile('D:\Research_Materials_PPK\Datos\Tamaños de grano_Mau y Pero\');
imds = imageDatastore(filenameTE, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);

Grandes = find(imds.Labels == 'Grandes', 1);
Finos = find(imds.Labels == 'Finos', 1);

figure
imshow(readimage(imds,Finos))

tbl = countEachLabel(imds)
totales=uint8(tbl.Count);
minix=100000;
maxix=0;
miniy=100000;
maxiy=0;
for k=1:totales(1)+totales(2)+totales(3)
    Imn=readimage(imds,double(k));
    [x, y, z]=size(Imn);
    if(x<minix)
        minix=x;
    end
    if(y<miniy)
        miniy=y;
    end

    if(x>maxix)
        maxix=x;
    end
    if(y>maxiy)
        maxiy=y;
    end

end
minix
miniy
```
the following characteristics were obtained:

  

Number of images
Desciption |Total images | 
--- | --- | 
Fine  | 87 | 
Medium| 47 | 
Big | 43 | 

RGM colour images
Size 1024X1280 


Next images are spplited into 4 smaller of size 512x512 and after data augemtation: rotate 90 degrees, rotate -90 degrees, flip up and flip left.
Also, images are resized to 0.5 and cropped to 224x224 pixel size
```
conteo=1;
for k=1:totales(1)
    Imn=readimage(imds,double(k));
    Imn=imresize(Imn,0.5);
    dir_outim = fullfile('D:\Research_Materials_PPK\Datos\Datos_prepared\Finos\');
    imagenS=Imn(1:224,1:224,:);
    Augmentation_2
    imagenS=Imn(1:224,257:257+223,:);
    Augmentation_2
    imagenS=Imn(257:257+223,1:224,:);
    Augmentation_2
    imagenS=Imn(257:257+223,257:257+223,:);
    Augmentation_2

end

Grandes = find(imds.Labels == 'Grandes', 1);

for k=Grandes:Grandes+totales(2)-1
    Imn=imresize(readimage(imds,double(k)),0.5);

    dir_outim = fullfile('D:\Research_Materials_PPK\Datos\Datos_prepared\Grandes\');
    imagenS=Imn(1:224,1:224,:);
    Augmentation_1
    imagenS=Imn(1:224,257:257+223,:);
    Augmentation_1
    imagenS=Imn(257:257+223,1:224,:);
    Augmentation_1
    imagenS=Imn(257:257+223,257:257+223,:);
    Augmentation_1

end

Medianos = find(imds.Labels == 'Medianos', 1)
for k=Medianos:Medianos+totales(3)-1
    Imn=imresize(readimage(imds,double(k)),0.5);
    
    dir_outim = fullfile('D:\Research_Materials_PPK\Datos\Datos_prepared\Medianos\');
    imagenS=Imn(1:224,1:224,:);
    Augmentation_1
    imagenS=Imn(1:224,257:257+223,:);
    Augmentation_1
    imagenS=Imn(257:257+223,1:224,:);
    Augmentation_1
    imagenS=Imn(257:257+223,257:257+223,:);
    Augmentation_1

end
```
A completely new set of 224x224 images were octained.

Desciption |Total images | 
--- | --- | 
Fine  | 2784 | 
Medium| 2256 | 
Big | 2064 | 



Example of Fine grain images

![F1](https://user-images.githubusercontent.com/44585823/171960413-7a700555-3582-4651-ab65-2879f1f3b50f.jpg)


Example of Medium grain images

![M1](https://user-images.githubusercontent.com/44585823/171960417-73d1ea0c-5914-41e5-97b4-182945859fbb.jpg)



Example of Big grain images

![FF1](https://user-images.githubusercontent.com/44585823/171960415-e0924914-ee5c-4de3-be25-2ca1a0ab8a67.jpg)




Next a Deep learning neural network was selected. From the image we can compare performance of different images and we can see





![networks_examples](https://user-images.githubusercontent.com/44585823/171680445-4b4e076c-6f8f-4df0-9a3f-3a589c283d70.png)

Relevant neural networks according to Accuracy, we selected RESNET50 for this project. The next code is to train the network. Data was splitted into training and validation sets.





outFolder='D:\Research_Materials_PPK\Datos\Datos_prepared';
imgDir = fullfile(outFolder);
imds = imageDatastore(imgDir, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

```
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7);
net=googlenet;

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
    
net= trainNetwork(augimdsTrain,lgraph,options);
```

Training process achieved an accuracy of 65%





![entrenamiento](https://user-images.githubusercontent.com/44585823/171682328-61d57794-ea9b-444e-8ac7-36a91193b92c.png)
