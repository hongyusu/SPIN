

%% 
% greedy algorithm running in negative feed mode
% input
%       gradient:       a 3*|E| column vector
% output
%       Ymax:           max gradient label
%       YmaxVal:        predicted value, not quite useful anywhere 
%
function [ Ymax, YmaxVal ] = global_greedy_labeling_directed(gradient)

    % edge of the graph
    global E;
    % how many exaples do we have here:
    %   m=1: optimization phase, we work on individual example at a time
    %   otherwise: inference phase, we work on many test items
    m = numel(gradient)/(3*size(E,1));
    
    YmaxVal=zeros(m,max(max(E)));
    
    yind=zeros(1,max(max(E)));
    yind(1)=1;
    y2y_local  = ones(size(yind,2))*(-1);
    uv_vp_nn = zeros(size(yind,2))*(-1);
    uv_vp_pn = zeros(size(yind,2))*(-1);
       
    %% iteration over all examples
    for i = 1:m
        % gradient for current examples: --,+-,++
        cur_g = gradient(:,(i-1)*size(E,1)+1:i*size(E,1));

        % assume we are now evaluating the part u-v-p of graph G,
        % where u is activated, v is under evaluation and is not activated, p could be many
        % now we want to compute the cost/gain of activating v (- -> +)
        % cost can be factorized as:
        %   upper stream, label change on uv from +- to ++
        %   lower stream, label change on vp from -- to +-
        
        % Step 1. work on upper stream: edge uv, label change from +- to ++,
        %   which is consider to be the local information (see suppl. of ICML paper)
        %   gain by (+-) -> (++)
        cur_g1 = cur_g(3,:)-cur_g(2,:); 
        for j = 1:size(yind,2)
            y2y_local(j,E(E(:,1)==j,2)) = cur_g1(E(:,1)==j);
        end
        
        % Step 2. work on lower stream: edge vp, label change from -- to +-,
        %   which is considered to be the global information added to local information (see suppl. of ICML paper)
        %   gain by (+-) -> (--)
        for j = 1:size(yind,2)
            for k = E(E(:,1)==j,2)'
                %u:j v:k
                uv_vp_nn(j,k) = sum(cur_g(1,E(:,2)==k&E(:,1)~=j));
                uv_vp_pn(j,k) = sum(cur_g(2,E(:,2)==k&E(:,1)~=j));
            end
        end
        y2y_global = uv_vp_pn - uv_vp_nn;
        
        % Step 3. combine upper and lower information to get the view on the global context
        y2y = y2y_local + y2y_global;
        
        % global greedy search: each time to pick up a node that maximize the global gain (see suppl. of ICML paper)
        %   in special case the objective function is submodular, the greedy search will have some optimization guarentee
        % stop criterion:
        %   1000 iteration or local gain cannot improve global objective
        n=1;
        while 1
            n = n+1;
            if n>1000
                break;
            end
            % choose the node that maximize the global gain
            [a,b] = find(y2y(yind==1,:)==max(max(y2y(yind==1,:))));
            a=a(1);
            b=b(1);
            % break if the the local will decrease global
            if y2y(a,b) < 0
                break
            end
            % update corresponding variables
            y2y(b,yind==1)=-1;
            y2y(a,b)=-1;
            yind(b)=n;
        end
        % activated nodes
        YmaxVal(i,:)=yind;
    end
    Ymax=(YmaxVal>0)*2-1;
    
    return
end







