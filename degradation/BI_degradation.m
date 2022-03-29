data_path='./VSRdataset/VID4/HR/';
output_path='./VSRdataset/VID4/BI/';
scale_factor    = 4;
imgDataDir  = dir(data_path);
for i = 1:length(imgDataDir)
    if(isequal(imgDataDir(i).name,'.')||... % 去除系统自带的两个隐文件夹
       isequal(imgDataDir(i).name,'..')||...
       ~imgDataDir(i).isdir)                % 去除遍历中不是文件夹的
           continue;
    end
    imgDir = dir([data_path imgDataDir(i).name]); 
    for j =1:length(imgDir)                 % 遍历所有图片
        if(isequal(imgDir(j).name,'.')||... % 去除系统自带的两个隐文件夹
           isequal(imgDir(j).name,'..'))                % 去除遍历中不是文件夹的
           continue;
        end
        imgDir(j).name
        img = im2double(imread([data_path imgDataDir(i).name '/' imgDir(j).name]));
        img = imresize(img, 1 / scale_factor, 'bicubic');
        if ~exist([output_path imgDataDir(i).name '/'], 'dir')
            mkdir([output_path imgDataDir(i).name '/']);
        end
        imwrite(img, [output_path imgDataDir(i).name '/' imgDir(j).name]);
    end
end
