
function run_SPINsdp(c,g,nb,penalty,type,filename)

    addpath('../spin/')
    addpath('./cvx/')
    cvx_setup;
    
    isTest = 0;

    % read input files X,Y,E,Yv,Yp
    if strcmp(filename(1:4),'meme')
        X=dlmread(sprintf('../memetracker/mt_%s.phix',filename)); % m * d feature matrix
        Y=dlmread(sprintf('../memetracker/mt_%s.Y',filename)); % m * k target matrix
        Yp=Y; % target matrix with time stamp 
        Y(Y>0)=1; % binary target matrix
        E=dlmread(sprintf('../memetracker/mt_%s.e',filename)); % edge of output graph
        E=E(:,1:2);
        Yv=dlmread(sprintf('../memetracker/mt_%s.Yv',filename)); % focual node
        YvSum=sum(Yv,2);
        Yv=Yv(YvSum>0,:);
        suffix=sprintf('%s_%s_%s_%s_%s_%s_SPINSDP', filename,c,g,nb,penalty,type);
    else
        X=dlmread(sprintf('../dblp/dblp.inp.%s.lda.phix',strrep(filename,'_','.')));
        Y=dlmread(sprintf('../dblp/dblp.inp.%s.lda.Y',strrep(filename,'_','.')));
        Yp=Y;
        Y(Y>0)=1;
        E=dlmread(sprintf('../dblp/dblp.inp.%s.e',strrep(filename,'_','.')));
        E=E(:,1:2);
        Yv=dlmread(sprintf('../dblp/dblp.inp.%s.lda.Yv',strrep(filename,'_','.')));
        YvSum=sum(Yv,2);
        Yv=Yv(YvSum>0,:);
        suffix=sprintf('%s_%s_%s_%s_%s_%s_SPINSDP', filename,c,g,nb,penalty,type)
    end

    % c,g,nb,penalty
    c=eval(c);
    g=eval(g);
    nb=eval(nb);
    penalty=eval(penalty);

    % select example
    Xsum=sum(X,2);
    X=X(Xsum~=0,:);
    Y=Y(Xsum~=0,:);
    Yp=Yp(Xsum~=0,:);
    Yv=Yv(Xsum~=0,:);

    %X=tfidf(X);

    % test
    if isTest
        X = X(1:20,:);
        Y = Y(1:20,:);
        Yv=Yv(1:20,:);
        Yp=Yp(1:20,:);
        size(X)
        size(Y)
        size(Yv)
        size(Yp)
    end

    % change Y from -1 to 0: labeling (0/1), for the sake of base learner svm, will change back in later section
    Y(Y==-1)=0;
    % stratified cross validation index
    nfold = 5;
    Ind = getCVIndex(Y,nfold);
    % get dot product kernel
    K = X * X';                         % dot product
    K = K ./ sqrt(diag(K)*diag(K)');    %normalization diagonal is 1

    %------------
    %
    % predict influence
    %
    %------------
    % set input parameters
    paramsIn.mlloss         = 0;        % assign loss to microlabels(0) edges(1)
    paramsIn.profiling      = 1;        % profile (test during learning)
    paramsIn.epsilon        = g;        % stopping criterion: minimum relative duality gap
    paramsIn.C              = c;        % margin slack
    paramsIn.max_CGD_iter   = 1;		% maximum number of conditional gradient iterations per example
    paramsIn.tolerance      = 1E-10;    % numbers smaller than this are treated as zero
    paramsIn.profile_tm_interval = 10;  % how often to test during learning
    paramsIn.maxiter        = 5;        % maximum number of iterations in the outer loop
    paramsIn.verbosity      = 1;
    paramsIn.debugging      = 3;
    paramsIn.nb             = nb;
    paramsIn.penalty        = penalty;
    paramsIn.type           = type;
    if isTest
        paramsIn.extra_iter     = 0;        % extra iteration through examples when optimization is over
    else
        paramsIn.extra_iter     = 0;        % extra iteration through examples when optimization is over
    end
    paramsIn.filestem       = sprintf('%s',suffix);		% file name stem used for writing output

    % to store results
    Ypred=[ones(size(Y,1),1),Y]; % one addition column for dummy node
    YpredVal=[ones(size(Y,1),1),Y];
    running_times=zeros(nfold,1);

    % N fold cross validation
    for k=1:nfold
        Itrain = find(Ind~=k);          % training index
        Itest  = find(Ind==k);
        gKx_tr = K(Itrain, Itrain);     % kernel
        gKx_ts = K(Itest,  Itrain)';
        gY_tr = Y(Itrain,:); gY_tr(gY_tr==0)=-1;    % training label
        gY_ts = Y(Itest,:); gY_ts(gY_ts==0)=-1;
        gYv_tr = Yv(Itrain,:);          % focal node
        gYv_ts = Yv(Itest,:);
        gYp_tr = Yp(Itrain,:);          % training label - with time stamp
        gYp_ts = Yp(Itest,:);
        
        % set input data
        dataIn.E = E;               % edge
        dataIn.Kx_tr = gKx_tr;      % kernel
        dataIn.Kx_ts = gKx_ts;
        dataIn.Y_tr = gY_tr;        % label
        dataIn.Y_ts = gY_ts;
        dataIn.Yv_tr = gYv_tr;      % focal node
        dataIn.Yv_ts = gYv_ts;
        dataIn.Yp_tr = gYp_tr;      % time stamp
        dataIn.Yp_ts = gYp_ts;

        % run influence prediction
        [rtn,ts_err] = SPINsdp(paramsIn,dataIn);
        
        % collect results
        load(sprintf('/var/tmp/Ypred_%s.mat', paramsIn.filestem));
        Ypred(Itest,:)=Ypred_ts;
        YpredVal(Itest,:)=Ypred_ts_val;
        running_times(k,1) = running_time;
        
        % get single round prediction
        %gY_ts=[ones(size(gY_ts,1),1),gY_ts];
        %[acc,vecacc,pre,rec,f1,auc1,auc2]=get_performance((gY_ts==1),(Ypred_ts==1));
        %[acc,vecacc,pre,rec,f1,auc1,auc2]
    end

    % finial prediction performance
    Y=[ones(size(Y,1),1),Y];

    [acc,vecacc,pre,rec,f1,auc1,auc2]=get_performance((Y==1),(Ypred==1));
    [acc,vecacc,pre,rec,f1,auc1,auc2]

    dlmwrite(sprintf('/var/tmp/Ypred_%s.mat',paramsIn.filestem),YpredVal);
    dlmwrite(sprintf('/var/tmp/T_%s.mat',paramsIn.filestem),running_times);
    system(sprintf('mv /var/tmp/Ypred_%s.mat ../results/; mv /var/tmp/T_%s.mat ../results/; rm /var/tmp/%s.log',paramsIn.filestem,paramsIn.filestem,paramsIn.filestem));
    
    if ~isTest
        exit
    end

end






