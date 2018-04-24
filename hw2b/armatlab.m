% M = csvread('ARMatrixControl.csv')
textscan('ARMatrxiControl.csv', '%s %s %s %s %s %s %s %s %s %s %s %s %s %s','delimiter', ',', 'EmptyValue', -Inf);


minSupport = 0.2;
minConfidence = 0.5;
nRules = 100;
fname = 'ControlRules';
for s = 1:size(textscan,1)
    labels{s} = ['Gene' num2str(s)];
end

% [Rules FreqItemsets] = findRules(Karate, minSup, minConf, nRules, sortFlag, labels, fname);
% disp(['See the file named ' fname '.txt for the association rules']);
% end
