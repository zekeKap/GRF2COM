folderPath="C:\Users\kapla\Desktop\ZEKE NUEMove\MachineLearningDataSet\Dataset2" %Change the folder directory 
files=dir(folderPath);
filePaths={};
GRFfiles={};
MKRfiles={};
Dataset={};
GRF={};
MKR={};
DataSetNames={};
% Loop through the files and get the full paths
for i = 1:length(files)
    % Exclude directories
    if ~files(i).isdir
        fullFilePath = fullfile(folderPath, files(i).name);
        filePaths{end+1} = fullFilePath;
    end
end
for j=1:length(filePaths)
    if contains(filePaths{j},"walkT")
    if contains(filePaths{j},"grf")
        GRFfiles{end+1}=filePaths{j};
    end
    end 
    if contains(filePaths{j},"walkT")
    if contains(filePaths{j},"mkr")
        MKRfiles{end+1}=filePaths{j};
    end
    end
    if contains(filePaths{j},"info")
        infoFile=filePaths{j};
    end
end
for j=1:length(GRFfiles)
    [path,name,ext]=fileparts(GRFfiles{j});
    name=erase(name,"grf");
    name=append(name,".mat");
    DataSetNames{j}=name;
end
for j=1:length(DataSetNames)
    columnname={'Time','Fx1','Fy1','Fz1','COPx1','COPy1','COPz1','Ty1',...
    	'Fx2','Fy2','Fz2','COPx2','COPy2','COPz2','Ty2'};
%     Dataset{1,1}=[columnname;table2cell(readtable(GRFfiles{j}))];
     Dataset{1,1}=[table2cell(readtable(GRFfiles{j}))];
    %Convert back to the table to add headers for the columns run code and
    %add to python code using pandas library in combo with scipy go for
    %specify not to use all marker data just the COG data
    %multiple regression to start with or polynomial regression
   columnname={'Time','R.ASISX','R.ASISY','R.ASISZ','L.ASISX','L.ASISY','L.ASISZ',...
   	'R.PSISX','R.PSISY','R.PSISZ','L.PSISX','L.PSISY','L.PSISZ','L.Iliac.CrestX',...
    'L.Iliac.CrestY','L.Iliac.CrestZ','R.Iliac.CrestX',	'R.Iliac.CrestY','R.Iliac.CrestZ',...
     'R.GTRX','R.GTRY','R.GTRZ','R.KneeX','R.KneeY','R.KneeZ','R.HFX','R.HFY','R.HFZ','R.TTX',...
 	'R.TTY','R.TTZ','R.AnkleX',	'R.AnkleY',	'R.AnkleZ',	'R.HeelX','R.HeelY'...
    'R.HeelZ','R.MT1X',	'R.MT1Y','R.MT1Z','R.MT5X','R.MT5Y','R.MT5Z','L.GTRX',...
    'L.GTRY','L.GTRZ','L.KneeX','L.KneeY','L.KneeZ','L.HFX','L.HFY','L.HFZ',...
    'L.TTX','L.TTY','L.TTZ','L.AnkleX','L.AnkleY','L.AnkleZ','L.HeelX',	'L.HeelY',...
    'L.HeelZ','L.MT1X','L.MT1Y','L.MT1Z','L.MT5X','L.MT5Y','L.MT5Z'};
%     Dataset{1,2}=[columnname;table2cell(readtable(MKRfiles{j}))];
 Dataset{1,2}=[table2cell(readtable(MKRfiles{j}))];
    save (DataSetNames{j},"Dataset");
    (j/(length(DataSetNames)))*100
end

% info=readtable(infoFile);
% info=table2cell(info);
% j=0;
% for j=1:length(info)
%     for t=1:29
%         Dataset{j,1}{1,t}=info{j,t};
%     end
% end
% save('DataMatFile.mat','Dataset','GRF','MKR');
% m=matfile('DataMatFile.mat','Writable',true);
% j=0;
% t=0;
% files={};
% filePaths={};
% Dataset={};
% count=1;
% for j=1:length(info)
%     for t=1:length(GRFfiles)
%         substring=info{j,1};
%         string=GRFfiles{t};
%         if contains(string,substring)
%             GRF{count,1}=table2cell(readtable(GRFfiles{t}));
%             clc
%             count=count+1;
%             (count/length(info))*100
%             %save('DataMatFile.mat','GRF');
%             break;
%         end
%     end
% end
% save('DataMatFile.mat','GRF','-v7.3');
% GRF={};
% 
% % for j=1:length(info)
% %     for t=1:length(GRFfiles)
% %         substring=info{j,1};
% %         string=GRFfiles{t};
% %         if contains(string,substring)
% %             Dataset=m.Dataset;
% %             Dataset{count,2}=table2cell(readtable(GRFfiles{t}));
% %             Dataset{count,3}=table2cell(readtable(MKRfiles{t}));
% %             save('DataMatFile.mat','Dataset');
% %             clc
% %             Dataset={};
% %             count=count+1;
% %             (count/length(info))*100
% %             break;
% %         end
% %     end
% % end
% % 
