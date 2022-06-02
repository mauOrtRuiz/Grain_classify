This project is for design a Neural network for Grain size classificarion. An input image is classified into 3 different clases: Fine, Medium and Big.
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
