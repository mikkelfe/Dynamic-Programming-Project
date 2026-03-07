clear all;

load fpathW1000th0rb7                       %%
LgT = [L;gT{1}(:,3)];
load fpathW1000th0                       %%
LgT2 = [L;gT{1}(:,3)];

figure(1);
plot([0:years],LgT2,'k-',[0:years],LgT,'k--','linewidth',1.4)
xlabel('Year \itt','fontsize',14)
ylabel('\itE[ L ]','fontsize',14)
ylim([500,1350])
%legend('Risk neutral    ','Risk averse    ',0)
legend('\itr_{b} = 6 %   ','\itr_{b} = 7 %   ',0)
set(gcf,'papersize',[3.5,2.5]);
set(gca,'fontsize',12,'xtick',[0:years],'ytick',[600,800,1000,1200,1400]);

