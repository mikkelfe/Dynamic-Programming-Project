clear all;

load figxpwdatath0rb7
XP2 = XP;
load figxpwdata

figure;
index = 5       % W is 1000,000 at 5th element in figxpwdata
W(index)
plot(P,XP(:,index),'k-',P,XP2(:,index),'k--','linewidth',1.5);
set(gca,'fontsize',12,'xtick',[1050:450:2850],'ytick',[-Lt,[(400-Lt):200:Lmax-Lt]])
set(gcf,'papersize',[3.5,2.5]);
ylabel('\itx_{t}*','fontsize',14)
%legend('Risk neutral  ','Risk averse  ',1)
%legend('Base var.  ','Higher var.  ',0)
legend('\itr_{b} = 6 %  ','\itr_{b} = 7 %  ',0)
xlim([Pmin Pmax+30])
ylim([-Lt-100 Lmax-Lt+100]);


clear all;

load figxpwdata10yr
XP10 = XP;
load figxpwdata
XP20 = XP;
load figxpwdata30yr
XP30 = XP;
load figxpwdata40yr
XP40 = XP;

figure;
index = 4
W(index)
plot(P,XP20(:,index),'k-',P,XP30(:,index),'k--','linewidth',1.5);
set(gca,'fontsize',12,'xtick',[Pmin:250:2210],'ytick',[-Lt,[(400-Lt):200:Lmax-Lt]])
set(gcf,'papersize',[3.5,2.5]);
ylabel('\itx_{t}*','fontsize',16)
%legend('Risk neutral  ','Risk averse  ',1)
%legend('Base var.  ','Higher var.  ',0)
%legend('\itr_{b} = 0.06  ','\itr_{b} = 0.07  ',0)
legend('20-year  ','30-year  ',0)
xlim([Pmin Pmax+30])
ylim([-Lt-100 Lmax-Lt+100]);


