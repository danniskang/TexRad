clear; close all;


%% parameters to be tuned
nReps       = 100;  % number of repetition
nCV         = 5;
% bagOfTrees  = [51,251,501,751,1001,2001];
bagOfTrees  = 251;

modalityName = {'T1','T2','ADC','DWI'};

FPR_range   = 0:0.02:1;
nFPR        = length(FPR_range);

%% 

for nn = 1 : length(bagOfTrees)
    
    nTrees = bagOfTrees(nn);
    
    for mm = 1 : length(modalityName)

        modality = modalityName{mm};

        for featType    = 1 : 3     % featType = 1: 'pre' / 2: 'mid' / 3: 'mid-pre'


            %% read xls files
            pcr         = xlsread(strcat('features_',modality),'Pre','T2:T137');
            pre_data    = xlsread(strcat('features_',modality),'Pre','B2:S137');
            mid_data    = xlsread(strcat('features_',modality),'Mid','B2:S137');
            del_data    = mid_data - pre_data;      % mid - pre

            if featType == 1
                featName= 'pre';        feat_raw= pre_data;
            elseif featType == 2
                featName= 'mid';        feat_raw= mid_data;
            elseif featType == 3
                featName= 'mid_pre';    feat_raw= del_data;
            end  


            rng(1);
            rnd_n   = randperm(2^16-1);
            rnd_seed= rnd_n(1:nReps);

            %% initialize for multiple repetitions
            AUC_saved       = zeros(nReps,1);       acc_saved       = zeros(nReps,1);
%             FPR_saved       = zeros(nFPR,nReps);    
            TPR_saved       = zeros(nFPR,nReps);
            Thr_saved       = zeros(nFPR,nReps);    confMat_saved   = cell(nReps,1);
            Y_val_saved     = zeros(length(pcr),nReps);
            Y_pred_saved    = zeros(length(pcr),nReps);
            score_saved     = zeros(length(pcr),nReps);
            
            for kk = 1 : nReps

                rng(rnd_seed(kk));
                cvObj = cvpartition(pcr,'k',nCV);

                %% initialize for cross-validation
                Y_pred_tmp    = [];
                Y_val_tmp     = [];
                score_tmp     = [];
                
                for ii = 1 : nCV

                    % partition the data based on cross-validation
                    trIdx   = cvObj.training(ii);
                    valIdx  = cvObj.test(ii);

                    feat_trn= feat_raw(trIdx,:);
                    Y_trn   = pcr(trIdx);

                    feat_val= feat_raw(valIdx,:);
                    Y_val   = pcr(valIdx);

                    %% feature normalization
                    meanTrain   = mean(feat_trn,1);
                    stdTrain    = std(feat_trn,1);
                    X_trn_cntr  = bsxfun(@minus,feat_trn,meanTrain);
                    X_trn       = bsxfun(@rdivide,X_trn_cntr,stdTrain);

                    X_tst_cntr  = bsxfun(@minus,feat_val,meanTrain);
                    X_val       = bsxfun(@rdivide,X_tst_cntr,stdTrain);

                    %% train random forest
                    model_rf        = TreeBagger(nTrees,X_trn,Y_trn,...
                        'OOBPred','off','OOBVarImp','on','NumPredictorsToSample','all');

%                     [featImp,featImpIdx]= sort(model_rf.OOBPermutedVarDeltaError,'descend');

                    [Y_pred,score]  = model_rf.predict(X_val);
                    Y_pred          = str2double(Y_pred);
                    
                    Y_pred_tmp    = [Y_pred_tmp; Y_pred];
                    Y_val_tmp     = [Y_val_tmp; Y_val];
                    score_tmp     = [score_tmp; score(:,2)];

                end

                acc         = sum(Y_val_tmp == Y_pred_tmp)/length(pcr);
                confMat     = confusionmat(Y_val_tmp,Y_pred_tmp,'order',[1,0]); 
                [FPR,TPR,Thr,AUC] = perfcurve(Y_val_tmp,score_tmp,'1','XVals',FPR_range,'UseNearest','off');
                
                AUC_saved(kk) = AUC;       
                acc_saved(kk) = acc;
%                 FPR_saved(:,kk) = FPR;    
                TPR_saved(:,kk) = TPR;    
                Thr_saved(:,kk) = Thr;
                confMat_saved{kk} = confMat;
                
                Y_val_saved(:,kk)   = Y_val_tmp;
                Y_pred_saved(:,kk)  = Y_pred_tmp;
                score_saved(:,kk)   = score_tmp;
            end
 
%             fname = strcat('results_multipleCV_',modality,'_',featName,'_',num2str(nTrees),'trees_ver3.mat');
%             save(fname,'confMat_saved','acc_saved','FPR','TPR_saved','Thr_saved','AUC_saved','Y_pred_saved','Y_val_saved','score_saved');
            
        end
    end
end

