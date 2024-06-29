function [B,Bt,A,A12,A13,A22,A23,C1,WB1,WB2] = train_OHPKL(X, Y, L,param,nbits,unlabel,B1,A,A12,A13,A22,A23,C1,WB1,WB2)
alpha = param.alpha;
beta = param.beta;
lambda = param.lambda;
theta = param.theta;
gamma = param.gamma;
[d, n] = size(X);
c = size(L,1);
n_unlabel=floor(unlabel* n);
Xu=X(:,(n - n_unlabel + 1) : n); Xl=X(:,1 : (n - n_unlabel));
Yu=Y(:,(n - n_unlabel + 1) : n); Yl=Y(:,1 : (n - n_unlabel));
% initialize Hash codes 
% sel_sample = Y(:,randsample(n, 2000),:);
[pcaW, ~] = eigs(cov(Y'), nbits);
V = pcaW'*Y;
Bt = sign(V);
Bt(Bt==0) = -1;
W1 = zeros(c,d);
W2 = zeros(c,d);
%Stage1
Ll = L(:,1 : (n - n_unlabel));
U = (NormalizeFea(Ll',1))';
Ll_=(NormalizeFea(Ll',1))';
XX = Xl * Xl';
YY = Yl * Yl';
WB1 = gamma*(WB1+ XX);
WB2 = gamma*(WB2+YY);
E1 =  Xl * Ll_'* Ll_;
E2 =  Yl * Ll_'* Ll_;
for iter = 1:param.iter   
    QQ1 = sqrt(sum( W1.*W1,2) + 1e-6);
    QQD1 = 1./QQ1;
    D11 = diag (QQD1);
    QQ2 = sqrt(sum( W2.*W2,2) + 1e-6);
    QQD2 = 1./QQ2;
    D22 = diag (QQD2);
    A = A+U * U';
    A12 = A12+U * Xl';
    A13 =  A13+ U * Ll_' *Ll_ * Xl';
%    WA1 = (eye(c)+ theta * A);
    WA1 = (eye(c)+ theta * A)\(lambda * D11);
    WC1 = (eye(c)+ theta * A)\(gamma*A12 + theta *A13);
    W1 = sylvester((WA1),(WB1),(WC1));
    
    A22 = A22+U * Yl';
    A23 = A23+ U * Ll_' *Ll_ * Yl';
%     WA2 = (eye(c)+ theta * A);
    WA2 = (eye(c)+ theta * A)\(lambda * D22);
    WC2 = (eye(c)+ theta * A)\( gamma*A22 + theta* A23);
    W2 = sylvester((WA2),(WB2),(WC2));
       U = (2 * eye(c)  + theta * W1 * XX * W1' + theta * W2 * YY * W2')...
\ (gamma*(W1 * Xl + W2 * Yl) + theta * W1 * E1 + theta * W2 * E2);
end

Lu = 0.5 * W1 * Xu + 0.5 * W2 * Yu;
Lu (Lu < 0) = 0;
L(:,(n - n_unlabel + 1) : n) = Lu;
% L(:,(n - n_unlabel + 1) : n) = 0;
%Stage2
L_ = (NormalizeFea(L',1))';

for iter = 1:param.iter 
    C1 = C1+V*L_';
    Bt = sign(nbits*alpha*C1*L_ + beta*V);
    
    K = beta*Bt + alpha*nbits*Bt*L_'*L_;
    K = K';
    Temp = K'*K-1/n*(K'*ones(n,1)*(ones(1,n)*K));
    [~,Lmd,QQ] = svd(Temp); clear Temp
    idx = (diag(Lmd)>1e-4);
    Q = QQ(:,idx); Q_ = orth(QQ(:,~idx));
    Pt = (K-1/n*ones(n,1)*(ones(1,n)*K)) *  (Q / (sqrt(Lmd(idx,idx))));
    P_ = orth(randn(n,nbits-length(find(idx==1))));
    V = sqrt(n)*[Pt P_]*[Q Q_]';
    V = V';  
end
    B = [B1,Bt];
end