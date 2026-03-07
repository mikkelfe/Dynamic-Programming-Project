clear all;
load figxpwdata
figure(2);
plot(P,XP(:,end),'k-',P,XP(:,1),'k--','linewidth',1.3);
set(gca,'fontsize',12,'xtick',[1050:450:2850],'ytick',[-600,-200:200:1400])
set(gcf,'papersize',[8.5,5.5]);
xlabel('\itP_{t}','fontsize',14)
%ylabel('Net Wealth: W  ($1000)','fontsize',10)
ylabel('\itx_{t}*','fontsize',14)
legend('\itW_{t} = $1,600,000  ','\itW_{t} = $400,000  ',1)
%text(1620,265,'\itW = 2300,000','fontsize',12);
%text(1450,-170,'\itW = 100,000','fontsize',12);
xlim([Pmin Pmax+50])
ylim([-650 1450]);
%--------------------------------------------------
[p,w]=meshgrid(P,W);
p = p'; w=w';

figure(1); 
plot3(p,w./1000,XP,'k-','linewidth',1.1);
view([10,11]);
set(gca,'fontsize',12,'xtick',[1050:450:2850],'ytick',[min(W)./1000:600:max(W)./1000],...
   'ztick',[-600,[-200:200:1400]]);
box on; set(gcf,'papersize',[8.5,5.5]);
xlabel('\itP_{\itt}','fontsize',14)
ylabel('\itW_{\itt} ($1000)','fontsize',14)
zlabel('\itx_{\itt}*','fontsize',14)
ylim([200,2000]);
zlim([-600 1500]);
%zoom reset;
%figure(1); mesh(p,w./1000,XP,w,'meshstyle','column','linewidth',1);box on; grid off;
%colormap(jet);
%set(gca,'fontsize',10,'xtick',[Pmin:250:2210],'ytick',[min(W)/1000:1000:max(W)/1000]);
%ylim([0 4000]);
%view([10,11]); box on; %set(gcf,'papersize',[8.5,5.5]);
%zlabel('\itx*','fontsize',12)
