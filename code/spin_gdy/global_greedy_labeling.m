

%% 
% greedy algorithm running in negative feed mode
% input
%       gradient:       a 5*|E| column vector
% output
%       Ymax:           max gradient label
%       YmaxVal:        predicted value, not quite useful anywhere 
%
function [ Ymax, YmaxVal ] = global_greedy_labeling(gradient)

    % edge of the graph
    global E;
    % how many exaples do we have here
    % m=1: optimization phase, we work on individual example at a time
    % otherwise: inference phase, we work on many test items
    m = numel(gradient)/(5*size(E,1));
    
    YmaxVal=zeros(m,max(max(E)));
    
    yind=zeros(1,max(max(E)));
    yind(1)=1;
    y2y_local  = ones(size(yind,2))*(-1);
    uv_vp_nn = zeros(size(yind,2))*(-1);
    uv_vp_pn = zeros(size(yind,2))*(-1);
       
    % iteration over all examples
    for i = 1:m
        % gradient for current examples 
        cur_g = gradient(:,(i-1)*size(E,1)+1:i*size(E,1));

        % assume u--v--p, u is activated, v is not activated, p could be many
        % now we want to compute the cost of activate v
        % cost can be factorized as upper stream and lower stream
        % step 1.upper stream: uv change from +- to ++
        cur_g1 = cur_g(4,:)-cur_g(3,:); %(+->+) - (+-)
        cur_g2 = cur_g(5,:)-cur_g(2,:); %(+<-+) - (-+)
        for j = 1:size(yind,2)
            y2y_local(j,E(E(:,1)==j,2)) = cur_g1(E(:,1)==j);
            y2y_local(j,E(E(:,2)==j,1)) = cur_g2(E(:,2)==j);
        end
        % step 2.lower stream: vp change from -- to +-
        % TODO: here is a mistake and should be corrected, second and forth row should be +=, the revision is not done in ICML paper experiments. It is supposed to imporve the results, though.
        for j = 1:size(yind,2)
            for k = E(E(:,1)==j,2)'
                %u:j v:k
%                 uv_vp_nn(j,k) = sum(cur_g(1,E(:,1)==k&E(:,2)~=j));
%                 uv_vp_nn(j,k) = sum(cur_g(1,E(:,2)==k&E(:,1)~=j));
                uv_vp_nn(j,k) = sum(cur_g(1,E(:,1)==k&E(:,2)~=j))+sum(cur_g(1,E(:,2)==k&E(:,1)~=j));
%                 uv_vp_pn(j,k) = sum(cur_g(3,E(:,1)==k&E(:,2)~=j));
%                 uv_vp_pn(j,k) = sum(cur_g(2,E(:,2)==k&E(:,1)~=j));
                uv_vp_pn(j,k) = sum(cur_g(3,E(:,1)==k&E(:,2)~=j))+sum(cur_g(2,E(:,2)==k&E(:,1)~=j));
            end
            for k = E(E(:,2)==j,1)'
                %u:j v:k
%                 uv_vp_nn(j,k) = sum(cur_g(1,E(:,1)==k&E(:,2)~=j));
%                 uv_vp_nn(j,k) = sum(cur_g(1,E(:,2)==k&E(:,1)~=j));
                uv_vp_nn(j,k) = sum(cur_g(1,E(:,1)==k&E(:,2)~=j))+sum(cur_g(1,E(:,2)==k&E(:,1)~=j));
%                 uv_vp_pn(j,k) = sum(cur_g(3,E(:,1)==k&E(:,2)~=j));
%                 uv_vp_pn(j,k) = sum(cur_g(2,E(:,2)==k&E(:,1)~=j));
                uv_vp_pn(j,k) = sum(cur_g(3,E(:,1)==k&E(:,2)~=j))+sum(cur_g(2,E(:,2)==k&E(:,1)~=j));
            end
        end
        y2y_global = uv_vp_pn - uv_vp_nn;
        % step 3.combine local and other to get global weight
        y2y = y2y_local + y2y_global;

        % greedy search: each time pick up a node that maximize the local gain
        % stop criterion: 1000 iteration or local gain cannot improve global objective
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
            if y2y(a,b) < -0
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
end







