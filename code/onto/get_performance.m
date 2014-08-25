

% label should be +1/0

function [acc,vecacc,pre,rec,f1,auc1,auc2,eacc,nbacc] = get_performance(Y,Ypred,Yp,YpredVal,E)

    acc = get_accuracy(Y,Ypred);
    vecacc = get_vecaccuracy(Y,Ypred);
    pre=get_precision(Y,Ypred);
    rec=get_recall(Y,Ypred);
    f1=get_f1(Y,Ypred);
    if nargin > 2
        auc1=0;
        auc2=0;
        [eacc,nbacc]=get_deacc(Yp,YpredVal,E);
    else
        auc1 = 0;
        auc2 = 0;
        eacc = 0;
        nbacc = 0;
    end
end

function [eacc,nbacc] = get_deacc(Yp, YpredVal, E)
    Yp(Yp<0)=0;
    YpredVal(YpredVal<0)=0;
    
    Yp=Yp(:,2:size(Yp,2));
    YpredVal=YpredVal(:,2:size(YpredVal,2));
    E=E(E(:,1)~=1,:)-1;
    Yh = Yp(:,E(:,1));
    Yt = Yp(:,E(:,2));
    YE = (double(Yh < Yt)*2-1) .* double((Yh ~=0) | (Yt ~=0));
    % pred
    Yh = YpredVal(:,E(:,1));
    Yt = YpredVal(:,E(:,2));
    YpredE = (double(Yh < Yt)*2-1) .* double((Yh ~=0) | (Yt ~=0));
    eacc = sum(sum(YpredE==YE))/size(YE,1)/size(YE,2);
    
    % nbacc
    nbacc=0;
    
    nbacc=zeros(size(Yp,1),1);
    for i=1:size(Yp,1)
        %[Yp(i,Yp(i,:)>0);YpredVal(i,Yp(i,:)>0)]
        nbacc(i)=corr(Yp(i,Yp(i,:)>0)',YpredVal(i,Yp(i,:)>0)','type','Kendall');
    end
    nbacc=mean(nbacc(~isnan(nbacc)));

end


function [auc1,auc2] = get_auc(Y,YpredVal)
    AUC=zeros(1,size(Y,2));
    for i=1:size(Y,2)
        [ax,ay,t,AUC(1,i)]=perfcurve(Y(:,i),YpredVal(:,i),1);
    end
    auc1=mean(AUC);
    [ax,ay,t,auc2]=perfcurve(reshape(Y,numel(Y),1),reshape(YpredVal,numel(YpredVal),1),1);
end

function [acc] = get_accuracy(Y,Ypred)

    acc=1-sum(sum(abs(Y-Ypred)))/size(Y,1)/size(Y,2);

end

function [vecacc] = get_vecaccuracy(Y,Ypred)

    vecacc=sum(Y~=Ypred,2);
    vecacc=sum((vecacc==0))/numel(vecacc);

end

function [f1] = get_f1(Y,Ypred)

    f1=(2*get_precision(Y,Ypred)*get_recall(Y,Ypred))/(get_precision(Y,Ypred)+get_recall(Y,Ypred));

end

function [tp] = get_tp(Y,Ypred)

    tp = Y + Ypred;
    tp=(tp==2);
    tp=sum(sum(tp));
    
end

function [fp] = get_fp(Y,Ypred)

    fp=Y-Ypred;
    fp=(fp==-1);
    fp=sum(sum(fp));

end

function [tn] = get_tn(Y,Ypred)

    tn=Y+Ypred;
    tn=(tn==0);
    tn=sum(sum(tn));

end

function [fn] = get_fn(Y,Ypred)

    fn=Y-Ypred;
    fn=(fn==1);
    fn=sum(sum(fn));

end

function [pre] = get_precision(Y,Ypred)

    pre=(get_tp(Y,Ypred))/(get_tp(Y,Ypred)+get_fp(Y,Ypred));

end

function [rec] = get_recall(Y,Ypred)

    rec=(get_tp(Y,Ypred))/(get_tp(Y,Ypred)+get_fn(Y,Ypred));

end
