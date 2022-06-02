# Grain_classify
A Neural network for Grain Classification Images. Images were captured by PhD Pedro Paramo with following characteristics:
Fine
Medium
Big
After a first inspection of data 
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

Size 1024X1280 


Next images are spplited into 4 smaller of size 512x512 and after data augemtation: rotate 90 degrees, rotate -90 degrees, flipp up and flip left follon total images.
Images are resized to 1/5 and cropped to 224x224 pixel size
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


Desciption |Total images | 
--- | --- | 
Fine  | 2784 | 
Medium| 2256 | 
Big | 2064 | 

Next a Deep learning neural network was selected.


![networks_examples](https://user-images.githubusercontent.com/44585823/171680445-4b4e076c-6f8f-4df0-9a3f-3a589c283d70.png)
Relevant neural networks according to Accuracy, Resnet50 was selected for this project
