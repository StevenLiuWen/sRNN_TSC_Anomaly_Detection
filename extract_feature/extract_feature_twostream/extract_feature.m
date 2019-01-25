clc;
clear;
addpath('../twostreamfusion/matconvnet/matlab');
addpath('../twostreamfusion/models');
addpath('../+dagnn/@DagNN')
run vl_setupnn

%%%%%%%%%%%%%%%%%%%%%%%%  setup here %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% only need to setup these 4 parameters
root = '../../';
dataset = 'shanghaitech';
res_type = '152';
gpu_id = 3; % id > 0


load_rgb_path = [root '/dataset/anomaly_detection/' dataset];
%load_optical_flow_path = '/home/luowx/datasets/avenue/testing/224/optical_flow/';
save_path = [root '/dataset/anomaly_detection/' dataset];
extract_layer = 'res5cx';
feature_dimension = 2048;
nstack = 10;
optical_flow_mean = 128;
clip_upper = 20;
clip_bottom = -20;
clip = true;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load the pre-trained CNN
net_rgb = load(['ucf101-img-resnet-' res_type '-split1.mat']);
net_rgb = dagnn.DagNN.loadobj(net_rgb.net) ;
net_rgb.mode = 'test' ;
net_rgb.conserveMemory = false; % note: important

% net_optical_flow = load('ucf101-TVL1flow-resnet-50-split1.mat');
% net_optical_flow = dagnn.DagNN.loadobj(net_optical_flow.net) ;
% net_optical_flow.mode = 'test' ;
% net_optical_flow.conserveMemory = false; % note: important

%set gpu
gpuDevice(gpu_id);
move(net_rgb,'gpu');
%move(net_optical_flow,'gpu');

% preprocess rgb
phases = {'/training', '/testing'};
for phase = phases
    phase = char(phase);
    load_rgb_path_new = [load_rgb_path phase '/frames/'];
    save_path_new = [save_path, phase '/224/features/twostream_res' res_type '_7x7'];
    
    dir_video = dir(load_rgb_path_new);
    dir_video = dir_video(3:end);
    for i = 1:length(dir_video)
        dir_image = dir([load_rgb_path_new dir_video(i).name]);
        dir_image = dir_image(3:end);
        num_frames = length(dir_image);
        % create hdf5
        h5create([save_path_new '/' dir_video(i).name '.h5'],'/rgb',[feature_dimension, 7, 7, num_frames],'Datatype','single');
        h5create([save_path_new '/' dir_video(i).name '.h5'],'/length',1,'Datatype','int32');
        features = zeros(num_frames, 7, 7, feature_dimension, 'single');
        for j = 1:num_frames
            im = imread([load_rgb_path_new dir_video(i).name '/' dir_image(j).name]);
            im_ = single(im) ;
            im_ = imresize(im_, net_rgb.meta.normalization.imageSize(1:2)) ;
            im_ = bsxfun(@minus, im_, net_rgb.meta.normalization.averageImage) ;
            net_rgb.eval({'input', im_}) ;
            feature = net_rgb.vars(net_rgb.getVarIndex(extract_layer)).value;
            features(j, :, :, :) = gather(feature);
            disp(['rgb : video ' num2str(i) ' / ' num2str(length(dir_video)) ' : frame ' num2str(j) ' / ' num2str(num_frames)]);
        end
        features = permute(features,[4,3,2,1]);
        h5write([save_path_new '/' dir_video(i).name '.h5'],'/rgb',features);
        h5write([save_path_new '/' dir_video(i).name '.h5'],'/length',int32(num_frames));
    end
end

% % preprocess optical flow
% dir_video = dir(load_optical_flow_path);
% dir_video = dir_video(3:end);
% for i = 1:length(dir_video)
%     optical_flow = h5read([load_optical_flow_path dir_video(i).name],'/optical_flow');
%     s = size(optical_flow);
%     num_frames = s(4) + 1;
%     % create hdf5
%     h5create([save_path '/' dataset '_' phrase '_' dir_video(i).name],'/optical_flow',[num_frames, feature_dimension],'Datatype','single');
%     features = zeros(num_frames, feature_dimension, 'single');
%     %normalize optical flow
%     for j = 1:num_frames - 1
%         for k = 1:2
%             if clip
%                 optical_flow(:,:,k,j) = max(optical_flow(:,:,k,j), clip_bottom);
%                 optical_flow(:,:,k,j) = min(optical_flow(:,:,k,j), clip_upper);
%             end
% 	    temp = optical_flow(:,:,k,j);
%             min_value = min(temp(:));
%             max_value = max(temp(:));
%             optical_flow(:,:,k,j) = bsxfun(@minus, optical_flow(:,:,k,j), min_value);
%             optical_flow(:,:,k,j) = bsxfun(@rdivide, optical_flow(:,:,k,j), max_value - min_value);
%             optical_flow(:,:,k,j) = optical_flow(:,:,k,j) * 255;
%         end
%     end
%     for j = 1:num_frames - nstack
%         im_ = optical_flow(:, :, :, j:j + nstack - 1);
%         im_ = permute(im_, [2, 1, 3, 4]);
%         s = size(im_);
%         im_ = reshape(im_, [s(1), s(2), s(3) * s(4)]);
%         im_ = imresize(im_, net_optical_flow.meta.normalization.imageSize(1:2)) ;
%         im_ = bsxfun(@minus, im_, optical_flow_mean) ;
%         net_optical_flow.eval({'input', im_}) ;
%         feature = net_optical_flow.vars(net_optical_flow.getVarIndex(extract_layer)).value;
%         features(j, :) = gather(feature(:));
%         disp(['optical_flow : video ' num2str(i) ' / ' num2str(length(dir_video)) ' : frame ' num2str(j) ' / ' num2str(num_frames)]);
%     end
%     for j = num_frames - nstack + 1:num_frames
%         features(j,:) = features(num_frames - nstack,:);
%     end
%     
%     h5write([save_path '/' dataset '_' phrase '_' dir_video(i).name],'/optical_flow',features);
% end
