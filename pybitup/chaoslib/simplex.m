function weight = simplex(data)

data = load(data);
A = data.V;
c = data.m;

options = optimoptions('linprog','Algorithm','dual-simplex','Display','off');
[nbrPts,nbrPoly] = size(A);

b = -ones(1,nbrPts);
A = [A,-2*eye(nbrPts)];
c = [-c,zeros(1,nbrPts)];
lb = [-1e5*ones(1,nbrPoly),zeros(1,nbrPts)];
ub = [1e5*ones(1,nbrPoly),ones(1,nbrPts)];

% Primal-dual simplex solution [primal,obj,exitflag,output,dual]

[~,~,~,~,dual] = linprog(c,[],[],A,b,lb,ub,[],options);
weight = dual.eqlin
end