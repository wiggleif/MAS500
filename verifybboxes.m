%% Setup the Import Options
opts = delimitedTextImportOptions("NumVariables", 8);

% Specify range and delimiter
opts.DataLines = [1, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["VarName1", "VarName2", "VarName3", "VarName4", "VarName5", "VarName6", "VarName7", "VarName8"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double"];
% opts = setvaropts(opts, [5, 6, 7, 8], "WhitespaceRule", "preserve");
% opts = setvaropts(opts, [5, 6, 7, 8], "EmptyFieldRule", "auto");
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
augset9fitted = readtable("/home/ossy/OneDrive/MAS500-Swarm/AI/TrainingData/AugmentedSet3/cameraset2_partiallycovered_fitted.csv", opts);


%% Clear temporary variables
% clear opts
%%
augTab1 = augset9fitted(:,1:4);
augTab2 = augset9fitted(:,5:8);
bb1_tab = table2array(augTab1);
bb2_tab = table2array(augTab2);
%%
for i=0:height(augTab1)-1
    i=i+1;
X1_Center_aug(i) = bb1_tab(i,1);
Y1_Center_aug(i) = bb1_tab(i,2);
X2_Center_aug(i) = bb2_tab(i,1);
Y2_Center_aug(i) = bb2_tab(i,2);
end

X1_Center_aug = X1_Center_aug';
Y1_Center_aug = Y1_Center_aug';
X2_Center_aug = X2_Center_aug';
Y2_Center_aug = Y2_Center_aug';
%%
X1_Center_augtable = array2table(X1_Center_aug);
X2_Center_augtable = array2table(X2_Center_aug);
Y1_Center_augtable = array2table(Y1_Center_aug);
Y2_Center_augtable = array2table(Y2_Center_aug);
w1_aug = varfun(@(var) var, augTab1(:,3));
h1_aug = varfun(@(var) var, augTab1(:,4));
w2_aug = varfun(@(var) var, augTab2(:,3));
h2_aug = varfun(@(var) var, augTab2(:,4));
%%
bboxtable1 = [X1_Center_augtable,Y1_Center_augtable,w1_aug,h1_aug];
bboxtable2 = [X2_Center_augtable,Y2_Center_augtable,w2_aug,h2_aug];
bboxarray1 = [table2array(bboxtable1)];
bboxarray2 = [table2array(bboxtable2)];
%% Write images with bounding boxes

      params = {'linewidth',2,'edgecolor','c'}; 
      syms f2;
      syms bbox2img;
      
for j = 0:length(bboxarray1)-1
    j = j+1;
    img = imread(sprintf('AugmentedSet3/AugmentedSet3_img_%d.png',j-1));
    f1 = @() rectangle('position', bboxarray1(j,:));
    bbox1img = insertInImage(img,f1,params);
        if ~any(isnan(bboxarray2(j,:))) == 1
            f2 = @() rectangle('position', bboxarray2(j,:))
            bbox2img = insertInImage(bbox1img,f2,params);
            imwrite(bbox2img, sprintf('VerifyAugSets/VerifyAugset3/verifyAugset3_%d.png',j-1));
%             imshow(bbox2img)
        else
            imwrite(bbox1img, sprintf('VerifyAugSets/VerifyAugset3/verifyAugset3_%d.png',j-1));
        end
end
    

