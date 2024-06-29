function [streamdata,streamdata_non,nstream,LTrain,L_tr,I_tr,T_tr,I_tr_non,T_tr_non,I_te_non,T_te_non] = predata_stream(I_tr,T_tr,L_tr,I_te,T_te,numbatch)

rand('seed',1);
anchors = 1000;
anchor_idx = randsample(size(I_tr,1), anchors);
XAnchors = I_tr(anchor_idx,:);
anchor_idx = randsample(size(T_tr,1), anchors);
YAnchors = T_tr(anchor_idx,:);
[I_tr_non,I_te_non]=Kernel_Feature(I_tr,I_te,XAnchors);
[T_tr_non,T_te_non]=Kernel_Feature(T_tr,T_te,YAnchors);

LTrain = L_tr;
row = size(LTrain,1);
data = randperm(row);
I_tr = I_tr(data,:);
T_tr = T_tr(data,:);
LTrain = LTrain(data,:);
L_tr = L_tr(data,:);

I_tr_non = I_tr_non(data,:);
T_tr_non = T_tr_non(data,:);

nstream = ceil(row/numbatch);
streamdata = cell(3,nstream);
streamdata_non = cell(3,nstream);
for i = 1:nstream-1
    start = (i-1)*numbatch+1;
    endl = i*numbatch;
    streamdata{1,i} = I_tr(start:endl,:);
    streamdata{2,i} = T_tr(start:endl,:);
    streamdata{3,i} = LTrain(start:endl,:);
    streamdata{4,i} = L_tr(start:endl,:);
    
    streamdata_non{1,i} = I_tr_non(start:endl,:);
    streamdata_non{2,i} = T_tr_non(start:endl,:);
    streamdata_non{3,i} = LTrain(start:endl,:);
    streamdata_non{4,i} = L_tr(start:endl,:);
end
start = (nstream-1)*numbatch+1;
streamdata{1,nstream} = I_tr(start:end,:);
streamdata{2,nstream} = T_tr(start:end,:);
streamdata{3,nstream} = LTrain(start:end,:);
streamdata{4,nstream} = L_tr(start:end,:);

streamdata_non{1,nstream} = I_tr_non(start:end,:);
streamdata_non{2,nstream} = T_tr_non(start:end,:);
streamdata_non{3,nstream} = LTrain(start:end,:);
streamdata_non{4,nstream} = L_tr(start:end,:);




