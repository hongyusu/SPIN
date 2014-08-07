function [Ymax,YmaxVal] = SDP_labeling_undirected(gradient)
    % gradient: a 5|E| column vector
    % Ymax:     max gradient labeling
    % YmaxVal:  predicted value
    
    global E;
    
    m = numel(gradient)/(5*size(E,1));
    YmaxVal=zeros(m,max(max(E)));  
    
    % adjacency matrix
    M_adj = full(sparse([E(:,1),E(:,2)],[E(:,2),E(:,1)],[ones(size(E,1),1),zeros(size(E,1),1)]));
    
    % loop throught examples
    for i = 1:m
        % current gradient for the example
        cur_g = gradient(:,(i-1)*size(E,1)+1:i*size(E,1)); 
        cur_g = cur_g - min(min(cur_g));    % all positive weight
        
        % S_pp, S_pn, S_nn, D
        S_pp = full(sparse([E(:,1),E(:,2)],[E(:,2),E(:,1)],[cur_g(4,:),cur_g(4,:)*0]));
        S_pn = full(sparse([E(:,1),E(:,2)],[E(:,2),E(:,1)],[cur_g(3,:),cur_g(4,:)*0]));
        S_np = full(sparse([E(:,1),E(:,2)],[E(:,2),E(:,1)],[cur_g(2,:),cur_g(4,:)*0]));
        S_nn = full(sparse([E(:,1),E(:,2)],[E(:,2),E(:,1)],[cur_g(1,:),cur_g(4,:)*0]));

        % sdp via CVX
        try
                Dsize = max(max(E)) + 1;
                cvx_begin sdp quiet
                variable D(Dsize,Dsize) symmetric
                minimize ( -sum(sum( (...
                        S_pp.*( 1 + repmat(D(1,2:Dsize),Dsize-1,1) + repmat(D(1,2:Dsize)',1,Dsize-1) + D(2:Dsize,2:Dsize)) + ...
                        S_pn.*( 1 + repmat(D(1,2:Dsize),Dsize-1,1) - repmat(D(1,2:Dsize)',1,Dsize-1) - D(2:Dsize,2:Dsize)) + ...
                        S_np.*( 1 - repmat(D(1,2:Dsize),Dsize-1,1) + repmat(D(1,2:Dsize)',1,Dsize-1) - D(2:Dsize,2:Dsize)) + ...
                        S_nn.*( 1 - repmat(D(1,2:Dsize),Dsize-1,1) - repmat(D(1,2:Dsize)',1,Dsize-1) + D(2:Dsize,2:Dsize)) ...
                        ) .*  M_adj)) );
                subject to
                        D >= 0;
                cvx_end
        catch err
                disp(err)
                D=zeros(max(max(E))+1,max(max(E))+1);
        end
        
        % imcomplete cholesky desomposition
        V = decomposition_chol(D);
        
        YmaxVal(i,:)=V';
    end
    Ymax = YmaxVal;
end

% imcomplete cholesky decompositon
function [V] = decomposition_chol(D)
    try
        pos = floor(size(D,1)/2);
        %pos=2;
        D=D([1,3:pos,2,(pos+1):size(D,1)],[1,3:pos,2,(pos+1):size(D,1)]);
        V = full(ichol(sparse(D))); % V\cdotV' = D
        V=V([1,pos,2:(pos-1),(pos+1):size(D,1)],:);
        V = V(2:size(V,1),:);
        
        V = sum(V .* repmat(V(1,:),size(V,1),1),2) ./ sqrt(sum(V.*V,2)); % dot product
        s = sort(V,'descend');
        %s'
        %sprintf('%d,%d,%d', size(V(V>1e-5),1), size(V(V>0),1), size(V(V>s(floor(size(s,1)*0.1))),1) )
        V = double(V > max(s(floor(size(s,1)*0.21)),0));
        %V = double(V > 0);
        V = V*2-1;
    catch err
        %disp(err)
        V = ones(1,size(D,1)-1);
    end 
end









