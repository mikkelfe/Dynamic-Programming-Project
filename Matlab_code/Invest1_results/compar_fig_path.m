clear all;

load pathw300L600
LgT3 = [L;gT{1}(:,3)];
load pathw700L600
LgT7 = [L;gT{1}(:,3)];
load pathw1000L600
LgT10 = [L;gT{1}(:,3)];
figure(1);
plot([0:years],LgT10,'k-.',[0:years],LgT7,'k--',[0:years],LgT3,'k-','linewidth',1.4)
ylabel('\itE[ L ]','fontsize',16)
ylim([350,1450])
set(gcf,'papersize',[8.5,5.5]);
set(gca,'fontsize',12,'xtick',[0:years],'ytick',[400,600,800,1000,1200,1400]);
legend('\itW_{0} = $1,000,000  ','\itW_{0} = $700,000  ','\itW_{0} = $300,000  ',4)

clear all

load pathw700L400
LgT4 = [L;gT{1}(:,3)];
load pathw700L600
LgT6 = [L;gT{1}(:,3)];
load pathw700L800
LgT8 = [L;gT{1}(:,3)];
figure(2);
plot([0:years],LgT8,'k-.',[0:years],LgT6,'k--',[0:years],LgT4,'k-','linewidth',1.4)
ylabel('\itE[ L ]','fontsize',16)
ylim([350,1450])
set(gcf,'papersize',[8.5,5.5]);
set(gca,'fontsize',12,'xtick',[0:years],'ytick',[400,600,800,1000,1200,1400]);
legend('\itL_{0} = 800','\itL_{0} = 600  ','\itL_{0} = 400  ',4)



clear all;

load pathw700L600th1                       %%
LgT = [L;gT{1}(:,3)];
load pathw700L600                       %%
LgT2 = [L;gT{1}(:,3)];
figure(3);
plot([0:years],LgT2,'k-',[0:years],LgT,'k--','linewidth',1.4)
ylabel('\itE[ L ]','fontsize',16)
ylim([350,1450])
legend('Risk neutral    ','Risk averse    ',0)
%legend('\itr_{b} = 0.06   ','\itr_{b} = 0.07   ',0)
set(gcf,'papersize',[3.5,2.5]);
set(gca,'fontsize',12,'xtick',[0:years],'ytick',[400,600,800,1000,1200,1400]);

clear all;
load pathw700L600                       
LgT = [L;gT{1}(:,3)];
load pathw700L600r37                       %%
LgT2 = [L;gT{1}(:,3)];
figure(4);
plot([0:years],LgT,'k-',[0:years],LgT2,'k--','linewidth',1.5)
ylabel('\itE[ L ]','fontsize',16)
ylim([350,1450])
%legend('Base var.    ','Higher var.    ',0)
legend('\itr_{b} = 0.06   ','\itr_{b} = 0.07   ',0)
set(gcf,'papersize',[3.5,2.5]);
set(gca,'fontsize',12,'xtick',[0:years],'ytick',[400,600,800,1000,1200,1400]);

clear all;

load pathw700L600T39
figure(5);
plot([0:years],[L;gT{3}(:,3)],'k-',[0:years],[L;gT{2}(:,3)],'k--','linewidth',1.5)
ylabel('\itE[ L ]','fontsize',16)
ylim([350,1450])
set(gcf,'papersize',[3.5,2.5]);
set(gca,'fontsize',12,'xtick',[0:years],'ytick',[400,600,800,1000,1200,1400]);
legend('20-year  ','30-year  ',0)
