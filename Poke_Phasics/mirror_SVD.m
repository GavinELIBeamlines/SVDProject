function mirror_SVD

list=ls('PHA*tif');

p=importdata(list(1,:));
crop=12;
p=p(crop+1:end-crop,crop+1:end-crop);
s=size(p);
poke=zeros(size(list,1),s(1)*s(2));

for ii=1:size(list,1)
    p=importdata(list(ii,:));
    p=double(p);
    x=importdata(['RAW Rep' sprintf('%02d',ii-1) '.txt'],'\t',3);
    p=p/65536*x.data(1);
    %     CROPPING
    p=p(crop+1:end-crop,crop+1:end-crop);
    poke(ii,:)=p(:)'-mean(p(:));
end


% imagesc(poke)
tic
[U S V]=svd(poke,'econ');
toc

U3=zeros(5,10,50);
U3(:)=U;
save('pokeSVD','U','S','V','U3','s','poke');