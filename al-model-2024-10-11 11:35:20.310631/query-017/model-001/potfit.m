clear;
clc;

% load data
inputdata = load("data.txt");
[ndata, ndim] = size(inputdata);

nbond = ndim - 1;

hiddenLayerSize = [15, 15];

Px = inputdata(:, 1: nbond);
Tx = inputdata(:, ndim);

P = Px'; %P nbond x ndata
T = Tx'; %T 1 x ndata

for i = 1: ndata
    if(T(1, i) > 2.0)
        ew(1, i) = 1.0;
    else
        ew(1, i) = 1.0;
    end
end

pra = minmax(P);
tra = minmax(T);

[m3, n3] = size(hiddenLayerSize);
net = feedforwardnet(hiddenLayerSize);
net.trainParam.showWindow=false; %do not show in -Xterm window
net.trainParam.showCommandLine=true; %generate commandline output

net.divideFcn= "dividerand";
net.divideMode = "sample";
net.divideParam.trainRatio = 0.90;
net.divideParam.valRatio = 0.05;
net.divideParam.testRatio = 0.05;
net.trainFcn = "trainlm";
net.performFcn = "mse";
net.trainParam.show = 20; % The result is shown at every 500th iteration (epoch) 
net.trainParam.lr = 0.001; % default 0.01
net.trainParam.epochs = 1000; % Max number of iterations 
net.trainParam.goal = 1e-9; % Error tolerance; stopping criterion
net.trainParam.max_fail = 6;
net.trainParam.time = inf;
net.trainParam.mu=1;
net.trainParam.mu_dec=0.8;
net.trainParam.mu_inc=1.5;

nfit = 1

for jfit=1:nfit
    disp(['Training ' num2str(jfit) '/' num2str(nfit)])

    tic;
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = true;

    rng('shuffle')
    net=init(net);
    % start training the network
    % 这里直接传入 P 和 T, feedforwardnet 会自动 scale
    [net,tr] = train(net,P,T,{},{},ew);

    trainingTime = toc;
    disp(['Training Time: ', num2str(trainingTime), ' s']);

    nwe=net.numWeightElements; %total number of non-linear paramters
    % 这里的 y 是 inverse scale 过后的
    y = net(P); %evaluate outputs on the fitted neural network

    trainout=y(tr.trainInd);
    validout=y(tr.valInd);
    testout =y(tr.testInd);

    trainTarget=T(tr.trainInd);
    validTarget=T(tr.valInd);
    testTarget=T(tr.testInd);

    %start outputing all parameters needed in later implementation in Fortran code.
    %first output some global parameters that define the network
    fid = fopen(['weights-', num2str(jfit), '.txt'], 'w');
    fprintf(fid, '%g\t', nbond); %number of input nodes
    fprintf(fid, '%g\t', n3); %number of hidden layers
    fprintf(fid, '%g\n', m3); %number of output nodes 
    fprintf(fid, '%g\t', hiddenLayerSize); %number of hidden nodes
    fprintf(fid, '\n');
    fprintf(fid, '%g\t', 1); %transfer function type, needs to be modified. 1='tansig',2='logsig',3='purelin'
    fprintf(fid, '%g\n', nwe); %number of total parameters including weights and biases
    fprintf(fid, '%20.16f\t', (pra(:,2)-pra(:,1))/2); %delta(input) for scale
    fprintf(fid, '%20.16f\n', (tra(:,2)-tra(:,1))/2); %delta(output) for scale
    fprintf(fid, '%20.16f\t', (pra(:,2)+pra(:,1))/2); %average(input) for scale
    fprintf(fid, '%20.16f\n', (tra(:,2)+tra(:,1))/2); %average(output) for scale

    %now output the weights

    %weights connecting input and the first hidden layer
    iw=net.IW{1,1}; % IW {3x1 cell} containing 1 input weight matrix
    [m, n] = size(iw); % m=50, n=6
    for i = 1 : m
            for j = 1 : n
            fprintf(fid, '%40.20f\n', iw(i,j));
            end
    end 

    %weights connecting hidden layers
    for i3=2:n3+1 % n3=2
        lw=net.LW{i3,i3-1}; % net.LW{2,1}, net.LW{3,2}, LW: {3x3 cell} containing 2 layer weight matrices
        [m, n] = size(lw); % 80x50, 1x80
        for i = 1 : m
            for j = 1 : n
                    fprintf(fid, '%40.20f\n', lw(i,j));
            end
        end 
    end

    %now output the biasess
    fid = fopen(['biases-', num2str(jfit), '.txt'], "w");
    for i3=1:n3+1 % n3 =2 b: {3x1 cell} containing 3 bias vectors
            b1=net.b{i3};
            [m, n] = size(b1); % 50x1; 80x1; 1x1
            for i = 1 : m
                    for j = 1 : n
                            fprintf(fid, '%40.20f\n', b1(i,j));
                    end
            end 
    end

    %now print the outputs and compare them with targets
    [m, n] = size(y);
    fid = fopen(['outputs-', num2str(jfit), '.txt'], "w");
    err=gsubtract(T,y);
    errmax = max(abs(err))
    for i = 1 : n
            fprintf(fid, '%15.8f%15.8f%15.8f', T(i),y(i),err(i));
            fprintf(fid, '\n');
    end

    %check the performace of the fitting
    disp("RMSE")
    totrmse = sqrt(mse(net,T,y))
    trainrmse = sqrt(mse(net,trainTarget,trainout))
    validrmse = sqrt(mse(net,validTarget,validout))
    testrmse = sqrt(mse(net,testTarget,testout))

    disp("MAE")
    totmae = mae(net,T,y)
    trainmae = mae(net,trainTarget,trainout)
    validmae = mae(net,validTarget,validout)
    testmae = mae(net,testTarget,testout)

    fprintf(fid,'%15.8f%15.8f%15.8f%15.8f%15.8f',totrmse,trainrmse,validrmse,testrmse,errmax);
    fprintf(fid,'\n');
    % net

    % genFunction(net, ['pesFcn_', num2str(jfit)], "MatrixOnly", "yes");
end
