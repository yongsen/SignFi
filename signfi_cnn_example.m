%% Yongsen Ma <yma@cs.wm.edu>
% Computer Science Department, The College of William & Mary
%
% This is an example for the following paper
% Yongsen Ma, Gang Zhou, Shuangquan Wang, Hongyang Zhao, and Woosub Jung. 2018.
% SignFi: Sign Language Recognition Using WiFi.
% Proc. ACM Interact. Mob. Wearable Ubiquitous Technol. 2, 1, Article 23 (March 2018), 21 pages.
% DOI: https://doi.org/10.1145/3191755

function [net_info, perf] = signfi_cnn_example(csi,label)
    % load('dataset_lab_276_dl.mat'); % load CSI and labels into workspace
    tic; % time of starting
    % prepare for training data
    csi_abs = abs(csi);
    csi_ang = angle(csi);
    % csi_ang = get_signfi_phase(csi);
    csi_tensor = [csi_abs,csi_ang];
    word = categorical(label);
    t0 = toc; % pre-processing time
    
    % [M,N,S]: CSI matrix for each instance
    % T: the total number of instances
    [M,N,S,T] = size(csi_tensor);
    Nw = 276; % number of classes
    
    rng(42); % For reproducibility
    n_epoch = 10;
    learn_rate = 0.01;
    l2_factor = 0.01;
    
    % Convolutional Neural Network settings
    layers = [imageInputLayer([M N S]);
              convolution2dLayer(4,4,'Padding',0);
              batchNormalizationLayer();
              reluLayer();
              maxPooling2dLayer(4,'Stride',4); 
              fullyConnectedLayer(Nw);
              softmaxLayer();
              classificationLayer()];
                         
    % get training/testing input
    K = 5;
    cv = cvpartition(T,'kfold',K); % 20% for testing
    k = 1; % for k=1:K
    trainIdx = find(training(cv,k));
    testIdx = find(test(cv,k));
    trainCsi = csi_tensor(:,:,:,trainIdx);
    trainWord = word(trainIdx,1);
    testCsi = csi_tensor(:,:,:,testIdx);
    testWord = word(testIdx,1);
    valData = {testCsi,testWord};
    
    % training options for the Convolutional Neural Network
    options = trainingOptions('sgdm','ExecutionEnvironment','cpu',...
                          'MaxEpochs',n_epoch,...
                          'InitialLearnRate',learn_rate,...
                          'L2Regularization',l2_factor,...
                          'ValidationData',valData,...
                          'ValidationFrequency',10,...
                          'ValidationPatience',Inf,...
                          'Shuffle','every-epoch',...
                          'Verbose',false,...
                          'Plots','training-progress');

    [trainedNet,tr{k,1}] = trainNetwork(trainCsi,trainWord,layers,options);
    t1 = toc; % training end time

    [YTest, scores] = classify(trainedNet,testCsi);
    TTest = testWord;
    test_accuracy = sum(YTest == TTest)/numel(TTest);
    t2 = toc; % testing end time
    
    % plot confusion matrix
%     ttest = dummyvar(double(TTest))';
%     tpredict = dummyvar(double(YTest))';
%     [c,cm,ind,per] = confusion(ttest,tpredict);
%     plotconfusion(ttest,tpredict);

    net_info = tr;
    perf = test_accuracy;
end
