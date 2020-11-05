% Data locations
frozenDir = 'C:\Users\jslee\MATLAB Drive\frozen\';
segDir = fullfile(frozenDir, 'nodule_seg');
nnDirBasenames = {'aug'}; % *_no_aug doesn't have inferences

dirNames = cell(size(nnDirBasenames));
for i = 1 : length(nnDirBasenames)
   dirNames{i} = fullfile(segDir, nnDirBasenames{i}); 
end
% Classes to plot
keepClasses = 2;

% Number of iterations to plot
%maxIters = 15000;
maxIters = []; % Unlimited

plot_frozen_curves(dirNames, keepClasses, maxIters)
