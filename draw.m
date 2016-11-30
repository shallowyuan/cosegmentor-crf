f1='loss7_1r1_t.txt';
f2='segloss_p5.txt'
%f1='loss6_1r1_t0.0001.txt';
%[~,class]=textread('classlist.txt','%d %s',-1);
loss=textread(f2,'%f',-1);
%loss=loss-max(0.02,min(loss)-0.02);
%subplot(2,1,1);
loss=reshape(loss(1:1620),20,[]);
loss=mean(loss);
plot(loss,'DisplayName','seg');
hold on
%loss2=textread(f1,'%f',-1);
%loss2=reshape(loss2(1:960),20,[]);
%loss2=mean(loss2);
%loss2=loss2-loss;
legend('show','Location','southeast')
%plot(loss2,'DisplayName','cls+seg');
hold off

saveas(gcf,'loss.jpg');
