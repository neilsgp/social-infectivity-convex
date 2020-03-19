clc;            % Initialization
clear;
D = 1000;       % Dimension of HP
U = zeros(D,9); % Initialization of Matrix U
V = zeros(D,9); % Initialization of Matrix V

for i = 1:9
    U(100*(i-1)+1:100*(i+1),i) = rand(200,1)/10;
    V(100*(i-1)+1:100*(i+1),i) = rand(200,1)/10;
end
A = U*V';
A = 0.8 * A./max(abs(eig(A)));

mu = rand(D,1)/D; % True Parameter 1000 dim vector
w = 1;            % w in the kernel function, to be 1 as stated above

num_samples = 250; % the number of sequences
nc = 100; % the maximum number of events per sequence
hp_samples = gen_hp_samples(w,mu,A, num_samples,nc); % generating samples

disp(hp_samples(1).timestamp);

%start optimization
num_iter_1 = 2;    % Number of Iteration of the First while loop (while k = 1,2 ...)
num_iter_2 = 7;    % Number of Iteration of the Second while loop (while not converge)
rho = 0.09;
thold = 1;         % thershold value

[x,y,Iteration_Err] = optimize_a_mu(hp_samples,D,w,rho,num_iter_1,num_iter_2,thold,A);

function [A_m,mu_m,Iteration_Err] = optimize_a_mu(hp_samples,D,w,rho,num_iter_1,num_iter_2,thold,real_A)
    % Input the training samples, parameters, number of iterations
    % and the real A for iterative learning.
    A_m = rand(D,D);          % Start with a random matrix to make optimization
    A_m = 0.8 * A_m./max(abs(eig(A_m)));  % Similar with the real A, scale the spectral radius of A_m to 0.8
    mu_m = rand(D, 1)./D;     % Start with a random mu value to make optimization as well
    UL = zeros(D,D);          % Matrix U's Parameter of Lowrank Property
    ZL = zeros(D,D);          % Matrix Z's Parameter of Lowrank Property
    US = zeros(D,D);          % Matrix U's Parameter of Sparse Property
    ZS = zeros(D,D);          % Matrix U's Parameter of Sparse Property
    %
    for i = 0:num_iter_1      % Outer while loop for algorithm 1 in the paper
		    rho = rho * 1.1;      % Iteratively change the value of rho each time
        for j = 0:num_iter_2  % Inner while loop for algorithm 1 in the paper
            % Main Algorithm mentioned above
            fprintf('No. %d outter while iteration | No. %d inner while iteration', i+1,j+1);
			    [A_m,mu_m,RelErr] = update_a_mu(A_m,mu_m,hp_samples,UL,ZL,US,ZS,w,rho,D,real_A);
            Iteration_Err(j+1,i+1) = RelErr;   % Capture the error during each iteration
        end
		    [s,v,d] = svd(A_m + US);     % Update ZL via SVD (Singular Value Decomposition)
		    v = v - thold/rho;
		    v(v < 0) = 0;                % Get ride of negatives
		    ZL = s * v * d';             % Update ZL (Z for LowRank)
		    UL = UL + (A_m - ZL);        % Update UL (U for LowRank)
		    tmp = (abs(A_m + US)-thold/rho);
		    tmp(tmp < 0) = 0;
		    ZS = (sign(A_m + US).*tmp);  % Update ZS (Z for Sparse)
		    US = US + (A_m-ZS);          % Update US (U for Sparse)
    end
end

function [A,mu,RelErr] = update_a_mu(A_m,mu_m,hp_samples,UL,ZL,US,ZS,w,rho,D,real_A)
    num_samples = length(hp_samples);
    %following the Algorthm 1: ADM4
    mu_numerator = zeros(D, 1);%init [D,1] all zeros; D -> dim of HP
    C = zeros(size(A_m));           % Initiation of the Matrix C (Equation 10 in the paper)
    A_Step = zeros(size(A_m)) + 2 * rho;
    B = zeros(size(A_m)) + rho*(UL-ZL) + rho*(US-ZS);% Add rho to B and its initialization (Equation 10 in the paper)
    for s = 1:num_samples           % Iterate through all the samples, 50,000
        timestamp = hp_samples(s).timestamp;% A vector, let's say 100 time stamps
        event = hp_samples(s).event;% A vector, each value a dim, same size as Time
        tc = timestamp(end);        % End time
%         disp(tc);
        nc = length(event);
        dt = tc - timestamp;        % dt for all every time step
%         disp(nc);
        for i = 1:nc                % Iterate through all the timestamps and events in ONE sample
            ui = event(i);          % Scalar
            ti = timestamp(i);      % Scalar
            int_g = kernel_int_g(dt, w);
            B(ui,:) = B(ui,:) + double(A_m(ui,:)>0).*repmat(int_g(i),[1,D]);
            pii = [];               % Initiation of the pii parameter is the Q objective function (Equation 8 in the paper)
            pij = [];               % Initiation of the pij parameter is the Q objective function (Equation 8 in the paper)
            ag_arr = [];            % Sum Auiuj*g(ti-tj)
            if i>1
                tj = timestamp(1:i-1);                        % Vector
                uj = event(1:i-1);                            % Vector
                ag_arr = A_m(uj, ui) .* kernel_g(ti - tj, w)';% Vector
            end
            pii = mu_m(ui)./(mu_m(ui) + sum(ag_arr));    % Calculation of pii in Equation 8 in the paper
%             disp(pii);
            if i>1
                pij = ag_arr./(mu_m(ui) + sum(ag_arr));  % Calculation of pij in Equation 8 in the paper
%                 disp(size(pij));
                if ~isempty(pij) && sum(pij(:))>0
%                     disp(length(uj));
                    for j = 1:length(uj)
                        uuj = uj(j);
                        C(uuj,ui) = C(uuj,ui) - pij(j,:);% Calculation of matrix C in Equation 10 in the paper
                    end
                end
            end
            mu_numerator(ui) = mu_numerator(ui) + pii;
        end                                           % for i = 1:nc (number of sequence in a sample)
    end                                               % for s = 1:num_samples (number of samples)
%     disp(B);
    mu = mu_numerator./(zeros(D, 1) + tc);            % Equation 9 in the paper
    A = (-B + sqrt(B.^2 - 4*A_Step.*C))./(2*A_Step);  % Equation 10 in the paper % Given AmatA = zeros(size(A_m)) + 2 * rho;
    
%     disp(A(500,1:5));
%     disp(A_m(500, 1:5));
    
    
    RelErr = real_err(real_A,A);                      % Storage of Iteration Relative Error
    fprintf('non-zero in mu= %d\n',nnz(mu));          % Testing of the sparse consition for mu
    fprintf('non-zero in C= %d\n',nnz(C));            % Testing of the sparse consition for C
    fprintf('non-zero in B = %d, non-zero in sqrt = %d\n',nnz(B),nnz(sqrt(B.^2 - 4*A_Step.*C)));
    fprintf('real error = %.4f, correlation = %.4f, #non-zero in A = %d\n\n',real_err(real_A,A),sum(abs(corr(real_A(:),A(:)))),nnz(A));
    A(isnan(A))=0;                                    % Refer to Equation 10 in the paper
    A(isinf(A))=0;
    %(A==0) = 0.00001;
end 

function err = real_err(A,A_m)
   % The calculation od relative error
   % Relative error is defined as the averaged
   % relative error between the estimated parameters
   % and the true parameters
   % i.e aij != 0 then err = err_1 or err = err_2 defined following.
    err_1 = abs(A - A_m) ./ A;
	    err_2 = abs(A - A_m);
	    err_1(isnan(err_1))=0;
    err_1(isinf(err_1))=0;
	    err = err_1 .* double(A~=0) + err_2.*double(A==0);
	    err = sum(err(:))/double(numel(A));
end

function G = kernel_int_g(dt, w)
G = 1 - exp(-w .* dt);
G(G<0) = 0;
end

function g = kernel_g(dt, w)
    % Kernel Function defined in the paper
    g = w * exp(-w .* dt);
    % Debugging to set value which has ripple effect
    g(g>1) = 0;
end

function hp_samples = gen_hp_samples(w,mu,A, num_samples,nc)
tic;
hp_samples = struct('timestamp', [],'event', []);
for n = 1:num_samples
    t=0;
    timestamp_and_event = [];
    lambdat = mu;                               % mu is the 1000 dim vector base intensity
    lambdat_sum = sum(lambdat);                 % Sum of all base intensities,used for gen next time-step
    while size(timestamp_and_event, 2) < nc
        rand_u = rand;
        dt = random('exp', 1/lambdat_sum);      % For generating the next time step
        lambda_ts = comp_lambda(t+dt, [], t, lambdat,w,mu,A);
        lambdats_sum = sum(lambda_ts);
        if rand_u > lambdats_sum / lambdat_sum  % Discard sample
            t = t+dt;
            lambdat = lambda_ts;
        else                                    % Keep sample
            u = rand * lambdats_sum;
            lambda_sum = 0;
            for d=1:length(mu)                  % d=1:1000
                lambda_sum = lambda_sum + lambda_ts(d);%sum over all lamda one by one
                if lambda_sum >= u              % Decide which dim event occurs
                    break;                      %end if
                end
            end
            lambdat = comp_lambda(t+dt, d, t, lambdat,w,mu,A);
            t = t+dt;
            timestamp_and_event = [timestamp_and_event,[t;d]];%append new sample
        end
        lambdat_sum = sum(lambdat);            % Time difference for next loop
    end                                        % While size(timestamp_and_event, 2)<nc
    hp_samples(n).timestamp = timestamp_and_event(1,:);
    hp_samples(n).event = timestamp_and_event(2,:);
    if mod(n, 10)==0                           % Print out the generation process
        fprintf('#samples=%d/%d, time=%.2fsec\n',n, num_samples, toc);
    end
end                                            %for n = 1:num_samples

function lambda = comp_lambda(cur_t, cur_event, last_t, lambdat,w,mu,A) 
lambda = mu + (lambdat - mu).*(exp(-w * (cur_t - last_t)));
if ~isempty(cur_event)%not empty, sum over all
    lambda = lambda + A(cur_event,:)';
end
end






end

