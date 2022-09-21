clear

filenameTE = fullfile('D:\300123\Documentos\Proyecto_GrainSize\Datos\');
imds = imageDatastore(filenameTE, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);

Grandes = find(imds.Labels == 'Grandes', 1);
Finos = find(imds.Labels == 'Finos', 1);

%figure
%montage(imds)

tbl = countEachLabel(imds)
totales=uint8(tbl.Count);

% dividir en imagenes 1024/2 by 1280/2

% aplicar data augmentation

% Solo hay 43 de medianos, es el numero menor.... reservar un 30% 8
% imagenes



% escenario #2  usar solo los parches 43X4 = 172  de los cuales separar 35

% AQUI SE PROCESAN LOS FINOS

Finos = find(imds.Labels == 'Finos', 1);

K1=0.5;
K2=0.5;

conteo=1;
%a = 1;
%b = totales(1);
%rFinosEp = uint8(double(b-a).*rand(16,1)+a);
bandera=0;
%save rFinosEp rFinosEp
load rFinosEp
for k=1:totales(1)
    
    if(sum(rFinosEp==k)==1)
        bandera=1;
    end
    if(k==75)
        bandera=0;
    end
    
    if(bandera==1)
        factor=0.5;
        
    
    
    Imna=readimage(imds,double(k));
    Imna=rgb2gray(imresize(Imna,0.5));
    Imna=double(Imna)./double(max(max(Imna)));
    y=histeq(Imna);
    y2 = imcomplement(edge(y,'canny'));
    ImnaInv=abs(1-Imna);
    ImnnaGamma=imadjust(Imna,[],[],0.4);
    Iprocessed=ImnnaGamma*K1+double(y2).*K2;
    Iprocessed2=uint8(Iprocessed*255);
    Imn=gray2rgb(Iprocessed2);
    
    
    %Imn=imresize(Imna,0.5);
    dir_outim = fullfile('D:\300123\Documentos\Proyecto_GrainSize\ValidationGrayImproved2\Finos\');
    imagenS=Imn(1:224,1:224,:);
    Augmentation_2
    imagenS=Imn(1:224,225:225+223,:);
    Augmentation_2
    imagenS=Imn(225:225+223,1:224,:);
    Augmentation_2
    imagenS=Imn(225:225+223,225:225+223,:);
    Augmentation_2
    end
    bandera=0;
    factor=0.5;
end
Grandes = find(imds.Labels == 'Grandes', 1);
%a=Grandes;
%b=Grandes+totales(2)-1;

%rGrandesEp = uint8(double(b-a).*rand(8,1)+a);
%save rGrandesEp rGrandesEp
load rGrandesEp
bandera=0;
for k=Grandes:Grandes+totales(2)-1
    factor=0.5;
    if(k==128)
        factor=0.22;
    end
    %Imna=readimage(imds,double(k));
    %Imn=imresize(Imna,factor);
    
    
    Imna=readimage(imds,double(k));
    Imna=rgb2gray(imresize(Imna,0.5));
    Imna=double(Imna)./double(max(max(Imna)));
    y=histeq(Imna);
    y2 = imcomplement(edge(y,'canny'));
    ImnaInv=abs(1-Imna);
    ImnnaGamma=imadjust(Imna,[],[],0.4);
    Iprocessed=ImnnaGamma*K1+double(y2).*K2;
    Iprocessed2=uint8(Iprocessed*255);
    Imn=gray2rgb(Iprocessed2);
    
    if(sum(rGrandesEp==k)==1)
        bandera=1;
    end
    if(bandera==1)
    dir_outim = fullfile('D:\300123\Documentos\Proyecto_GrainSize\ValidationGrayImproved2\Grandes\');
    imagenS=Imn(1:224,1:224,:);
    Augmentation_1
    imagenS=Imn(1:224,225:225+223,:);
    Augmentation_1
    imagenS=Imn(225:225+223,1:224,:);
    Augmentation_1
    imagenS=Imn(225:225+223,225:225+223,:);
    Augmentation_1
    end
    bandera=0;
    factor=0.5;

end

Medianos = find(imds.Labels == 'Medianos', 1)

%a=Medianos;
%b=Medianos+totales(3)-1;

%rMedianosEp = uint8(double(b-a).*rand(8,1)+a);
%save rMedianosEp rMedianosEp
load rMedianosEp

bandera=0;
for k=Medianos:Medianos+totales(3)-1
    factor=0.5;
    if(k>169)
        factor=0.20;
    end
    if(sum(rMedianosEp==k)==1)
        bandera=1;
    end
    if(bandera==1)
        
        
   % Imna=readimage(imds,double(k));
   % Imn=imresize(Imna,factor);
    
    Imna=readimage(imds,double(k));
    Imna=rgb2gray(imresize(Imna,0.5));
    Imna=double(Imna)./double(max(max(Imna)));
    y=histeq(Imna);
    y2 = imcomplement(edge(y,'canny'));
    ImnaInv=abs(1-Imna);
    ImnnaGamma=imadjust(Imna,[],[],0.4);
    Iprocessed=ImnnaGamma*K1+double(y2).*K2;
    Iprocessed2=uint8(Iprocessed*255);
    Imn=gray2rgb(Iprocessed2);
    
    dir_outim = fullfile('D:\300123\Documentos\Proyecto_GrainSize\ValidationGrayImproved2\Medianos\');
    imagenS=Imn(1:224,1:224,:);
    Augmentation_1
    imagenS=Imn(1:224,225:225+223,:);
    Augmentation_1
    imagenS=Imn(225:225+223,1:224,:);
    Augmentation_1
    imagenS=Imn(225:225+223,225:225+223,:);
    Augmentation_1
    end
    bandera=0;
   factor=0.5;
    

end


