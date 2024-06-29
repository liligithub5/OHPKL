
clc;
clear;  
load mirflickr25k.mat
% load nus-wide.mat
% load IJiapr-tc12.mat    
unlabel = 0.2;                                       
numbatch = 2000;
run = 1;
bits = [16,32,64,128];
%% Preprocessing data 
I_te  = bsxfun(@minus, I_te, mean(I_tr, 1)); 
I_tr = bsxfun(@minus, I_tr, mean(I_tr,1));
T_te  = bsxfun(@minus, T_te, mean(T_tr, 1));
T_tr = bsxfun(@minus, T_tr, mean(T_tr,1));
[streamdata,streamdata_non,nstream,LTrain,L_tr,I_tr,T_tr,I_tr_non,T_tr_non,I_te_non,T_te_non] = predata_stream(I_tr,T_tr,L_tr,I_te,T_te,numbatch);
for i = 1:length(bits) 
     for j=1:run  
        [B_I,B_T,tB_I,tB_T] = main_OHPKL(streamdata_non, I_te_non, T_te_non, bits(i),unlabel);
        Dhamm = hammingDist(tB_I, B_T)';    
        [~, HammingRank]=sort(Dhamm,1);
        mapII = map_rank(L_tr,L_te,HammingRank); 
        Dhamm = hammingDist(tB_T, B_I)';    
        [~, HammingRank]=sort(Dhamm,1);
        mapTI = map_rank(L_tr,L_te,HammingRank); 
        map(j, 1) = mapII(end);
        map(j, 2) = mapTI(end);      
    end
    fprintf('\n bits = %d\n', bits(i));
    fprintf('OHPKL %d bits -- Image_to_Text_MAP: %.4f ; Text_to_Image_MAP: %.4f ; \n',bits(i),mean(map( : , 1)),mean(map( : , 2)));
 end

