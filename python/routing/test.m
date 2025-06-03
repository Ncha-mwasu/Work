function [CH_F,net] = HEED_algo(R,D,p,pMin,E,Emax,net,cost)
    N = size(net,2); % number of nodes
    CH_prop = max([p*(E./Emax);pMin*ones(1,N)]);
    CH_index = find(p*(E./Emax) >= CH_prop);
    CH = zeros(1,N);
    CH(CH_index) = 1;
    for i=1:N
        if (CH(i) == 1)
            net(3,i) = i;
        else
            min_cost = min(cost(CH==1));
            net(3,i) = find(cost==min_cost,1);
        end
    end
    CH_F = find(CH==1);
end
