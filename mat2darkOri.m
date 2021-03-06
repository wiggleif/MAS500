%% Import data from text file
% Script for importing data from the following text file:
%
%    filename: /home/ossy/OneDrive/MAS500-Swarm/AI/TrainingData/augset9/augset9fitted.csv
%    
%    The file is attached at the same github branch, but will have to
%    adjust the filepath to make it work on your computer

%% Setup the Import Options
opts = delimitedTextImportOptions("NumVariables", 8);

% Specify range and delimiter
opts.DataLines = [1, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["VarName1", "VarName2", "VarName3", "VarName4", "VarName5", "VarName6", "VarName7", "VarName8"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double"];
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
augset9fitted = readtable("/home/ossy/OneDrive/MAS500-Swarm/AI/TrainingData/AugmentedSet7/augmentedset7_fitted_fixed.csv", opts);


%% Clear temporary variables
% clear opts
%%
augTab1 = augset9fitted(:,1:4);
augTab2 = augset9fitted(:,5:8);
bb1_tab = table2array(augTab1);
bb2_tab = table2array(augTab2);
%% Transfor (x,y) from TL corner to center
for i=0:height(augTab1)-1
    i=i+1;
X1_Center_aug(i) = bb1_tab(i,1)+bb1_tab(i,3)/2;
Y1_Center_aug(i) = bb1_tab(i,2)+bb1_tab(i,4)/2;
X2_Center_aug(i) = bb2_tab(i,1)+bb2_tab(i,3)/2;
Y2_Center_aug(i) = bb2_tab(i,2)+bb2_tab(i,4)/2;
end

X1_Center_aug = X1_Center_aug';
Y1_Center_aug = Y1_Center_aug';
X2_Center_aug = X2_Center_aug';
Y2_Center_aug = Y2_Center_aug';
%%
%% Transform dat-file format into acceptable yolo format
X1_Center_yolo = X1_Center_aug/640;
X1_Center_yolo = array2table(X1_Center_yolo);
Y1_Center_yolo = Y1_Center_aug/480;
Y1_Center_yolo = array2table(Y1_Center_yolo); 

w1_aug_yolo = varfun(@(var) var/640, augTab1(:,3));
h1_aug_yolo = varfun(@(var) var/480, augTab1(:,4));

X2_Center_yolo = X2_Center_aug/640;
X2_Center_yolo = array2table(X2_Center_yolo);
Y2_Center_yolo = Y2_Center_aug/480;
Y2_Center_yolo = array2table(Y2_Center_yolo);

w2_aug_yolo = varfun(@(var) var/640, augTab2(:,3));
h2_aug_yolo = varfun(@(var) var/480, augTab2(:,4));
%% Result:
yolo_aug_Table1 = [X1_Center_yolo,Y1_Center_yolo,w1_aug_yolo,h1_aug_yolo];
yolo_aug_Table2 = [X2_Center_yolo,Y2_Center_yolo,w2_aug_yolo,h2_aug_yolo];
firstcolumn = zeros(height(yolo_aug_Table1),1);
yolo_aug_array1 = [firstcolumn,table2array(yolo_aug_Table1)];
yolo_aug_array2 = [firstcolumn,table2array(yolo_aug_Table2)];
%% Write label data to txt files (to correct folder)
for j = 0:length(yolo_aug_array1)-1
    j = j+1;
    [index,data] = fopen(sprintf('/home/ossy/OneDrive/MAS500-Swarm/AI/TrainingData/AugmentedSet7/label7_fixed/AugmentedSet7_img_%d.txt',j-1),'wt');
    test_aug_array1 = yolo_aug_array1(j,:);
    fprintf(index, '%6.0f %1.5f %1.5f %1.5f %1.5f\n',test_aug_array1);
        if ~any(isnan(yolo_aug_array2(j,:))) == 1
       test_aug_array2 = yolo_aug_array2(j,:);
       fprintf(index, '%6.0f %1.5f %1.5f %1.5f %1.5f',test_aug_array2);
             else 
        end
    fclose(index);
end
