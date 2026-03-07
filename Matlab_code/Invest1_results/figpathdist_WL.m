clear all;
load figpathdistdata;

syears = [1,5,10];

syear = 3;        %choose 1,2, or 3 (upto length(syears))

%minW1 = min(gs{syear}(:,2));
%maxW1 = max(gs{syear}(:,2));
minW1 = -140000;
maxW1 = 5460000;
range1 = maxW1 - minW1;
gap1 = range1/40;
edge1 = [minW1:gap1:maxW1];
fp1 = histc(gs{syear}(:,4),edge1);
figure; 
bar(edge1/1000,fp1/500,1,'c');
xlim([-1000 5500])
ylim([0,0.13])
set(gcf,'papersize',[8.5,5.5]);
set(gca,'fontsize',14,'layer','top')%,'ytick',[0:0.02:0.23]);
xlabel(['\itW' ' in Year ' num2str(syears(syear)), '  ($1000)'],'fontsize',16)
ylabel('Probability','fontsize',16)


edge2 = [0,Lmin:50:Lmax];
fp2 = histc(gs{syear}(:,3),edge2);
figure;
bar(edge2,fp2/500,1,'c');
xlim([-150,2150]);
ylim([0,0.53])
set(gcf,'papersize',[8.5,5.5]);
set(gca,'fontsize',14,'layer','top','xtick',[0,400,800,1200,1600,2000]);
xlabel(['\itL' ' in Year ' num2str(syears(syear))],'fontsize',16)
ylabel('Probability','fontsize',14)