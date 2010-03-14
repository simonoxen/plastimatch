%% Conjugate gradient with soft line search.
%% See Eqn 4.11 
%%   in Frandsen, Jonasson, Nielsen, and Tingleff's booklet:
%%   "Unconstrained Optimization" 
%%   http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/3217/pdf/imm3217.pdf
x0 = [1; 0];
delta = 1;
max_its = 1;
alg = 'conjugate-gradient-1';

alpha = 1;
x = x0;
[f,g] = rosenbrock (x);
disp (sprintf ('%10g [%10g %10g] [%10g %10g], %10g', ...
               f, x(1), x(2), g(1), g(2), alpha));
gamma = 0;
hcg = [0;0];
for it = 1:max_its
    
    %% Compute search direction
    hprev = hcg;
    hcg = -g + gamma*hprev;

    %% Check for wrong direction
    if (g' * hcg > 0)
        hcg = -hcg;
    end
    
    %% Initialize line search
    line_search_complete = 0;
    a = 0; b = 1;
    amax = 10;
    rho = 0.2; beta = 0.5;
    phip0 = - hcg' * hcg;         % phi'(0)
    lsg = beta * phip0;           % gamma in alg 2.27

    %% Perform line search part 1
    for k=1:5

        %% Evaluate function
        x1 = x + b * hcg;
        [f1,g1] = rosenbrock (x1);

        %% Update interval
        lambda_b = f + rho * phip0 * b;
        phipb = - g1' * hcg;

        disp (sprintf ('(1) %9f %9f %9f %9f',...
                       f1, lambda_b, phipb, lsg));
        if (f1 < lambda_b && phipb < lsg)
            a = b;
            b = b * 2;
        end
    end

    %% Perform line search part 2
    alpha = b;
    for k=1:5

        %% Evaluate function
        x1 = x + b * hcg;
        [f1,g1] = rosenbrock (x1);

        kk kk kk
        
    end
    
    gprev = g;
    f = f1;g = g1;x = x1;

    %% Fletcher-Reeves
    gamma = (g' * f) / (gprev' * gprev);
        
    %% Polak-Ribiere
    gamma = ((g-gprev)' * g) / (gprev' * gprev);
        
end
