function WFR_disp(data)
% Data is a stracture with variables from the PokeSVD script.

X=[1;1;1;1;1]*[0:13.5:121.5];
Y=[13.5*[1 1 1 1 1]', [0 27 54 81 108]']*[[1 0 1 0 1 0 1 0 1 0];[1 1 1 1 1 1 1 1 1 1]];
Xm=0:13.5:121.5;
Ym=0:13.5:121.5;
[Xm,Ym]=meshgrid(Xm,Ym);
n=3;
Zm=griddata(X,Y,data.U3(:,:,n),Xm,Ym);

pha_list=ls('PHA*.int');
raw_list=ls('RAW*.txt');

% pcolor(3/2*Xm,3/2*Ym,Zm);
% hold on
% plot(3/2*X,3/2*Y,'or')


% hold on
% plot3(3/2*X,3/2*Y,'or')
close all;

fig = uifigure('Position',[50 50 500 240],'name','Select a mode');

% global H;

global TakeModes;
global ShotNO;
TakeModes=12;
ShotNO=10;

H.label1 = uilabel(fig,...
    'Position',[80 205 300 15],...
    'Text','Mode View: 3');

H.label2 = uilabel(fig,...
    'Position',[80 135 300 15],...
    'Text','Current Shot: 10');
H.label3 = uilabel(fig,...
    'Position',[80 65 300 15],...
    'Text',['Subtracting ' num2str(TakeModes) ' modes']);

H.sld = uislider(fig,'Position',[50 190 400 55],'Value',0,...
    'ValueChangedFcn',@(sld,event) updateMode(event,Xm,Ym,data,X,Y,H),...
    'Limits',[0 49]);
% H.sld2 = uislider(fig,'Position',[50 120 400 55],'Value',ShotNO,...
%     'ValueChangedFcn',@(sld,event) updateShot(event,pha_list,raw_list,data,H),...
%     'Limits',[0 size(pha_list,1)]);
H.sld2 = uislider(fig,'Position',[50 120 400 55],'Value',ShotNO,...
    'ValueChangedFcn',@(sld,event) updateShot(event,pha_list,raw_list,data,H),...
    'Limits',[0 50]);
H.sld3 = uislider(fig,'Position',[50 50 400 55],'Value',TakeModes,...
    'ValueChangedFcn',@(sld,event) updateResiduals(event,pha_list,raw_list,data,H),...
    'Limits',[0 49]);


f=figure;
set(f,'units','centimeters');
s=2*[21 10.5];
set(f,'position',[3 5 s]);
set(f,'papersize',s);

% subplot(2,3,1)
% surf(Xm,Ym,Zm);
% title('Mirror modes')
% xlabel('X, mm');
% ylabel('Y, mm');
% view(-10,82)
%
% subplot(2,3,2)
% % plot of singluar values
% semilogy(diag(data.S),'-x')
% title('Singular values')
% xlabel('Mode')
% ylabel('Singular values')
%
% subplot(2,3,3)
% % plot of detector modes
% % pcolor()

end


function updateMode(event,Xm,Ym,data,X,Y,H)
n = round(event.Value)+1;

subplot(3,3,1)
hold off;
Zm=griddata(X,Y,data.U3(:,:,n),Xm,Ym);
surf(Xm,Ym,Zm);
view(-10,82)
hold on
plot3(X,Y,data.U3(:,:,n),'or')
title('Mirror modes')
xlabel('X, mm');
ylabel('Y, mm');

subplot(3,3,2)
% plot of singluar values
m=diag(data.S);
hold off
semilogy(m,'-x');
hold on;
semilogy(n,m(n),'or')
title('Singular values')
xlabel('Mode')
ylabel('Singular values')

subplot(3,3,3)
% plot of detector modes
M=zeros(data.s);
M(:)=data.V(:,n);
imagesc(M);
title('Detector modes')
xlabel('X, arb.u.');
ylabel('Y, arb.u.');
H.label1.Text=['Selected mode: ' [num2str(n-1)]];
end


function updateShot(event,pha_list,raw_list,data,H)
n = round(event.Value)+1;
global ShotNO;
ShotNO=n;
% H.label2.Text=['Current shot: ' pha_list(n)];
FigUpdate(pha_list,raw_list,data,H)

end


function updateResiduals(event,pha_list,raw_list,data,H)
n = round(event.Value)+1;
global TakeModes
TakeModes=n;
H.label3.Text=['Subtracting ' num2str(TakeModes) ' modes'];
FigUpdate(pha_list,raw_list,data,H)

end

function FigUpdate(pha_list,raw_list,data,H)
global TakeModes
global ShotNO;
n=ShotNO;
pocet_modu=rank(data.S);
% pha_name=pha_list(n,:);
% raw_name=raw_list(n,:);
% d=importdata(pha_name,'\t',7);
% p=d.data;
% crop=12;
% p=p(crop+1:end-crop,crop+1:end-crop);
% s=size(p);
% ptv=importdata(raw_name,'\t',3); ptv=ptv.data(1);
% p=p/(2^16)*ptv;
% p=p-mean(p(:));

% % !!!!!!!!!!!! manual cropping)
% p(:,1:1)=[];
% p(:,end-14:end)=[];
p=load('deltas','W2');
p=p.W2/600; %normalizace na 600 kroku
s=size(p);
subplot(3,3,4)
imagesc(p);
text(0.15,0.85,['RMS = ' num2str(std(p(:)),3) ' \lambda'],'units','normalized')
text(0.15,0.72,['PtV = ' num2str(max(p(:))-min(p(:)),3) ' \lambda'],'units','normalized')

cmin=min(min(p));
cmax=max(max(p));
title('Measured Phase, units=waves')
xlabel('X, pixels')
ylabel('Y, pixels')
colorbar
colormap jet
% H.label2.Text=['Shot ' pha_name(1:23)];



res=p(:);
for ii=1:TakeModes
    x=data.V(:,ii);
    res=res-dot(res,x)*x;
end
Res=zeros(s);
Res(:)=res;
Modefit=p-Res;

subplot(3,3,5)
imagesc(Modefit,[cmin cmax])
text(0.15,0.85,['RMS = ' num2str(std(Modefit(:)),3) ' \lambda'],'units','normalized')
text(0.15,0.72,['PtV = ' num2str(max(Modefit(:))-min(Modefit(:)),3) ' \lambda'],'units','normalized')
title('Reconstructed with modes, units=waves')
xlabel('X, pixels')
ylabel('Y, pixels')
colorbar

subplot(3,3,6)
imagesc(Res,[cmin cmax]);
text(0.15,0.85,['RMS = ' num2str(std(Res(:)),3) ' \lambda'],'units','normalized')
text(0.15,0.72,['PtV = ' num2str(max(Res(:))-min(Res(:)),3) ' \lambda'],'units','normalized')
title('Residual phase, units=waves')
xlabel('X, pixels')
ylabel('Y, pixels')
colorbar

subplot(3,3,7)
% Plot of modal decomposition
x=(p(:)-res)';
cor=(x*data.V(:,1:pocet_modu));
x=res';
resid=(x*data.V(:,1:pocet_modu));
hold off
% yyaxis left
bar(cor',1,'green','stacked');
hold on;
bar(resid',1,'red','stacked');
xlim([0, pocet_modu+1])
hold off


% yyaxis right
% dg=diag(data.S);
% dg=dg(1:rank(data.S));
% cor=cor./dg';
% resid=resid./dg';
% 
% bar(cor+resid,1,'blue');
% ylim auto

title('Mirror mode decomposition')
xlabel('Modes')
ylabel('Projection')

subplot(3,3,8)
% estimate modal decomposition
% x=(p(:)-res)';
% cor=(x*data.V(:,1:pocet_modu));
% title('Corrected modes')
% xlabel('Modes')
% ylabel('Projection')

subplot(3,3,9)
% estimate modal decomposition
% x=res';
% resid=(x*data.V(:,1:pocet_modu));
% title('Residual modes')
% xlabel('Modes')
% ylabel('Projection')

end

