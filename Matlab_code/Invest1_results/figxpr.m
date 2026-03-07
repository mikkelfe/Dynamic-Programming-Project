clear all;
load figxprdata
figure(2);
plot(P,XP(:,end),'k-',P,XP(:,1),'k--','linewidth',1.3);
set(gca,'fontsize',12,'xtick',[1050:450:2850],'ytick',[-600,-200:200:600])
set(gcf,'papersize',[8.5,5.5]);
xlabel('\itP_{t}','fontsize',14)
%ylabel('Net Wealth: W  ($1000)','fontsize',10)
ylabel('\itx_{t}*','fontsize',14)
legend('\itR_{t} = 430  ','\itR_{t} = 330  ',1)
%text(1620,265,'\itR = 430','fontsize',12);
%text(1450,-170,'\itR = 330','fontsize',12);
xlim([Pmin Pmax+50])
ylim([-650 650]);
%--------------------------------------------------
[p,r]=meshgrid(P,R);
p = p'; r=r';

figure(1); 
plot3(p,r,XP,'k-','linewidth',1.1);
view([6,14]);
set(gca,'fontsize',12,'xtick',[1050:450:2850],'ytick',[min(R):50:max(R)],...
    'ztick',[-600,[-200:200:1400]]);
box on; set(gcf,'papersize',[8.5,5.5]);
xlabel('\itP_{\itt}','fontsize',14)
ylabel('\itR_{\itt}','fontsize',14)
zlabel('\itx_{\itt}*','fontsize',14)
ylim([280 500]);
zlim([-600 700]);
