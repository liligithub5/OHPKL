function [Bi_Ir,Bt_Tr,Bi_Ie,Bt_Te] = main_OHPKL(streamdata, I_te, T_te, bits,unlabel)
%% --------mirflickr/NUS-WIDE/IJiapr-tc -------- %%
param.lambda = 1e-1; param.theta = 1e-3; %Stage1
param.alpha = 1; param.beta =  1e-2; %Stage2
param.sigma = 1e-1;  param.gamma = 1;
param.iter  = 5;

nstream = size(streamdata,2);
for chunki = 1:nstream 
    Itrain = streamdata{1,chunki}';  Ttrain = streamdata{2,chunki}';Ltrain = streamdata{3,chunki}';
    if chunki == 1
        t0 = cputime;
      [B,A,A12,A13,A22,A23,C1,WB1,WB2] = train_OHPKL0(Itrain, Ttrain, Ltrain,param, bits,unlabel); 
        D1 = Itrain*Itrain';
        D2 = Ttrain*Ttrain';
        F1 = B*Itrain';
        F2 = B*Ttrain';
       Time_online(chunki,1) = cputime-t0;
       clear t0;
    else             
       t1 = cputime;
       [B,Bt,A,A12,A13,A22,A23,C1,WB1,WB2] = train_OHPKL(Itrain,Ttrain,Ltrain,param,bits,unlabel,B,A,A12,A13,A22,A23,C1,WB1,WB2);
       D1 = D1+Itrain*Itrain';
       D2 = D2+Ttrain*Ttrain';
       F1 = F1 + Bt*Itrain';
       F2 = F2 + Bt*Ttrain';
       Time_online(chunki,1) = cputime-t1;
        clear t1;
    end
end
    PI = F1*pinv(D1+param.sigma*eye(size(D1,1)));
    PT = F2*pinv(D2+param.sigma*eye(size(D2,1)));
    Bt_Tr = compactbit(B' >= 0);
    Bi_Ir = compactbit(B' >= 0);
    Yi_te = sign((bsxfun(@minus,I_te*PI' , mean(I_te*PI',1))));
    Yt_te = sign((bsxfun(@minus,T_te*PT' , mean(T_te*PT',1))));
    Bt_Te = compactbit(Yt_te >= 0);
    Bi_Ie = compactbit(Yi_te >= 0);
end
