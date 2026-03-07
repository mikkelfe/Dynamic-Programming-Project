clear all;

load figxpwdatath0var56R
XP2 = XP;
load figxpwdata

figure;
index = 5       % W is 1000,000 at 5th element in figxpwdata
W(index)
plot(P,XP(:,index),'k-',P,XP2(:,index),'k--','linewidth',1.5);
set(gca,'fontsize',12,'xtick',[1050:450:2850],'ytick',[-Lt,[(400-Lt):200:Lmax-Lt]])
set(gcf,'papersize',[3.5,2.5]);
xlabel('\itP_{t}','fontsize',14)
ylabel('\itx_{t}*','fontsize',14)
legend('Less risky  ','More risky  ',1)
%legend('Base var.  ','Higher var.  ',0)
%legend('\itr_{b} = 6 %  ','\itr_{b} = 7 %  ',1)
xlim([Pmin Pmax+50])
ylim([-400 1600]);

