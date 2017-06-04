function [net, info] = cnn_cifar(varargin)

%Demonstrates ResNet (with preactivation) on:
%CIFAR-10 and CIFAR-100 (tested for depth 164)

% run(fullfile(fileparts(mfilename('fullpath')), ...
%   '..', 'matconvnet','matlab', 'vl_setupnn.m')) ;

opts.modelType = 'oded' ;
opts.normImdb = true;
opts.depth=164;
opts.GPU=1;
opts.batchSize=128;
opts.weightDecay=0.00001;
opts.momentum=0.9;
opts.resConn = 1;
opts.Nclass=10;
opts.colorSpace = 'rgb';
opts.resType = '131';
opts.learningRate = [0.01*ones(1,10) 0.001*ones(1,2) 0.01*ones(1,2) 0.001*ones(1,5) 0.01*ones(1,2) 0.001*ones(1,20)] ;
opts.learningRate = [0.001*ones(1,2) opts.learningRate  0.005*ones(1,20) 0.001*ones(1,20)] ;
% opts.learningRate = [0.0001*ones(1,4) 0.0001*ones(1,12) 0.001*ones(1,5)  0.001*ones(1,91-5-12-4) 0.0005*ones(1,30) 0.0001*ones(1,30)] ;
% opts.learningRate = linspace(0.01,0.0001,150);
% opts.k = [64 128 256 512];
opts.k = [32 64 128 256];
opts.usePad = true;
opts.solver = @solver.adagrad;
opts.solverOpts = opts.solver();
% opts.solverOpts.rho = 0.8;


opts.useQR = true;

[opts, varargin] = vl_argparse(opts, varargin) ;

datas='cifar';
colorStr = '';
if ~strcmp(opts.colorSpace, 'rgb')
    colorStr = ['-' opts.colorSpace];
end

switch opts.modelType
    case 'res'
        opts.expDir = sprintf(['data/%s_%d-%s-D%d-R%d-T%s' colorStr],datas, opts.Nclass, opts.modelType,opts.depth,opts.resConn,opts.resType);
    case 'fractal'
        opts.expDir = sprintf(['data/%s_%d-%s-D%d-R%d-T%s' colorStr],datas, opts.Nclass, opts.modelType,opts.depth,opts.resConn,opts.resType);
    case 'assaf'
        padStr = '';
        if ~opts.usePad
            padStr = '_noPad';
        end
        opts.expDir = sprintf(['data/%s_%d_%s_%d_%d_%d_%d' colorStr padStr],datas, opts.Nclass, opts.modelType,opts.k);
    case 'oded'
        padStr = '';
        if ~opts.usePad
            padStr = '_noPad';
        end
        qrStr = '';
        if opts.useQR
            qrStr = '_useSVD';
        end
        opts.expDir = sprintf('data/%s_%d_%s',datas, opts.Nclass, opts.modelType);
        opts.expDir = [opts.expDir sprintf('_%d',opts.k) colorStr padStr qrStr '_largeSkip_addFC'];
end

[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile(vl_rootnn, 'data', datas) ;
switch opts.Nclass
    case 10
        opts.dataDir = 'D:\Datasets\DetectionTrainDB\Cifar-10';
    case 100
        opts.dataDir = 'D:\Datasets\DetectionTrainDB\Cifar-100';
end

% gpuDevice(1)

opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.networkType = 'dagnn' ;
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = [opts.GPU]; end;

% -------------------------------------------------------------------------
%                                                    Prepare model and data
% -------------------------------------------------------------------------

switch opts.modelType
  case 'res'
    net = cnn_resnet_preact('modelType', opts.modelType,'depth',opts.depth, 'resConn', opts.resConn, 'Nclass', opts.Nclass, 'resType', opts.resType, 'colorSpace', opts.colorSpace);  
  case 'fractal'
    net = cnn_fractalnet('depth',opts.depth, 'Nclass', opts.Nclass, 'colorSpace', opts.colorSpace);      
  case 'assaf'
    net = cnn_assafnet('Nclass', opts.Nclass, 'colorSpace', opts.colorSpace, 'k', opts.k, 'usePad', opts.usePad);      
  case 'oded'
    net = cnn_oded_net('Nclass', opts.Nclass, 'colorSpace', opts.colorSpace, 'k', opts.k, 'usePad', opts.usePad);
  otherwise
    error('Unknown model type ''%s''.', opts.modelType) ;
end


allBlockTypes = cellfun( @(c) class(c), {net.layers.block}, 'UniformOutput', false);
lossBlockInds = ismember( allBlockTypes, {'dagnn.Loss','dagnn.PDist'});


net.meta.trainOpts.learningRate=opts.learningRate; %update lr
net.meta.trainOpts.batchSize = opts.batchSize; %batch size
net.meta.trainOpts.weightDecay = opts.weightDecay; %weight decay
% net.meta.trainOpts.momentum = opts.momentum ;
net.meta.trainOpts = rmfield(net.meta.trainOpts,'momentum');
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate); %update num. ep.

if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath) ;
else
    if opts.Nclass==10 && strcmp(datas,'cifar')
        imdb = getCifar10Imdb(opts) ;
        mkdir(opts.expDir) ;
        save(opts.imdbPath, '-struct', 'imdb') ;
    else
        imdb = getCifar100Imdb(opts) ;
        mkdir(opts.expDir) ;
        save(opts.imdbPath, '-struct', 'imdb') ;
    end
end

net.meta.classes.name = imdb.meta.classes(:)' ;
net.meta.normalization.mean_image = imdb.meta.normalization.mean_image;
net.meta.normalization.std_image = imdb.meta.normalization.std_image;

% -------------------------------------------------------------------------
%                                                                     Train
% -------------------------------------------------------------------------

[net, info] = cnn_train_dag(net, imdb, getBatch(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 3)) ;

% -------------------------------------------------------------------------
function fn = getBatch(opts)
% -------------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
    error('The simplenn structure is not supported for the ResNet architecture');
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus),...
                   'modelType',opts.modelType,...
                   'numClass',opts.Nclass ...
                    ) ;
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if rand > 0.5, images=fliplr(images) ; end
images=cropRand(images) ; %random crop for all samples
if opts.numGpus > 0
  images = gpuArray(images) ;
end
switch opts.modelType
    case 'oded'
        inputs2 = {};
        if size(imdb.images.labels,1) > 1
            labels2 = imdb.images.labels(2,batch) ;
            inputs2 = {['label_class',num2str(2),'_',num2str(20)], labels2};
        end
        inputs = [{'input', images, ['label_class',num2str(1),'_',num2str(opts.numClass)], labels} inputs2];
        
    otherwise
        inputs = {'input', images, 'label', labels} ;
end

% -------------------------------------------------------------------------
function imdb = getCifar10Imdb(opts)
% -------------------------------------------------------------------------
unpackPath = fullfile(opts.dataDir, 'cifar-10-batches-mat');
files = [arrayfun(@(n) sprintf('data_batch_%d.mat', n), 1:5, 'UniformOutput', false) ...
  {'test_batch.mat'}];
files = cellfun(@(fn) fullfile(unpackPath, fn), files, 'UniformOutput', false);
file_set = uint8([ones(1, 5), 3]);

if any(cellfun(@(fn) ~exist(fn, 'file'), files))
  url = 'http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz' ;
  fprintf('downloading %s\n', url) ;
  untar(url, opts.dataDir) ;
end

data = cell(1, numel(files));
labels = cell(1, numel(files));
sets = cell(1, numel(files));
for fi = 1:numel(files)
  fd = load(files{fi}) ;
  data{fi} = permute(reshape(fd.data',32,32,3,[]),[2 1 3 4]) ;
  labels{fi} = fd.labels' + 1; % Index from 1
  sets{fi} = repmat(file_set(fi), size(labels{fi}));
end

set = cat(2, sets{:});

switch opts.colorSpace
    case 'rgb'
        % nothing
    case 'gray'
        data = cellfun(@(x) cellfun(@rgb2gray, num2cell(x,[1 2 3]), 'UniformOutput', false), data, 'UniformOutput', false);
        data = cellfun(@(x) cat(4,x{:}), data, 'UniformOutput', false);
    case 'yuv'
        data = cellfun(@(x) cellfun(@rgb2ycbcr, num2cell(x,[1 2 3]), 'UniformOutput', false), data, 'UniformOutput', false);
        data = cellfun(@(x) cat(4,x{:}), data, 'UniformOutput', false);      
end

data = single(cat(4, data{:}));

%pad the images to crop later
data = padarray(data,[4,4],128,'both');


%remove mean
data_tr = data(:,:,:, set == 1);
data_tr = reshape(permute(data_tr,[1 2 4 3]),[],1,size(data_tr,3));

meanCifar = mean(data_tr,1);
data = bsxfun(@minus, data, meanCifar);

%divide by std
stdCifar = std(data_tr,0,1);
data = bsxfun(@rdivide, data, stdCifar) ;

clNames = load(fullfile(unpackPath, 'batches.meta.mat'));

imdb.images.data = data ;
imdb.images.labels = single(cat(2, labels{:})) ;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = clNames.label_names;
imdb.meta.normalization.mean_image = meanCifar;
imdb.meta.normalization.std_image = stdCifar;

% -------------------------------------------------------------------------
function imdb = getCifar100Imdb(opts)
% -------------------------------------------------------------------------
unpackPath = fullfile(opts.dataDir, 'cifar-100-matlab');
files{1} = fullfile(unpackPath, 'train.mat');
files{2} = fullfile(unpackPath, 'test.mat');
%files{3} = fullfile(unpackPath, 'meta.mat');
file_set = uint8([1, 3]);

if any(cellfun(@(fn) ~exist(fn, 'file'), files))
  url = 'http://www.cs.toronto.edu/~kriz/cifar-100-matlab.tar.gz' ;
  fprintf('downloading %s\n', url) ;
  untar(url, opts.dataDir) ;
end

data = cell(1, numel(files));
labels = cell(1, numel(files));

sets = cell(1, numel(files));
for fi = 1:numel(files)
  fd = load(files{fi}) ;
  data{fi} = permute(reshape(fd.data',32,32,3,[]),[2 1 3 4]) ;
  labels{fi} = fd.fine_labels' + 1; % Index from 1
  labels{fi}(2,:) = fd.coarse_labels' + 1; % Index from 1
  sets{fi} = repmat(file_set(fi), 1, size(labels{fi},2));
end

set = cat(2, sets{:});
data = single(cat(4, data{:}));

%pad the images to crop later
data = padarray(data,[4,4],128,'both');

if opts.normImdb
    % remove mean
    r = data(:,:,1,set == 1);
    g = data(:,:,2,set == 1);
    b = data(:,:,3,set == 1);
    meanCifar = [mean(r(:)), mean(g(:)), mean(b(:))];
    data = bsxfun(@minus, data, reshape(meanCifar,1,1,3));

    %divide by std
    stdCifar = [std(r(:)), std(g(:)), std(b(:))];
    data = bsxfun(@times, data,reshape(1./stdCifar,1,1,3)) ;
else
    meanCifar = zeros(1,3);
    stdCifar = ones(1,3);
end

clNames = load(fullfile(unpackPath, 'meta.mat'));

imdb.images.data = data ;
imdb.images.labels = single(cat(2, labels{:})) ;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = clNames.fine_label_names;
imdb.meta.normalization.mean_image = meanCifar;
imdb.meta.normalization.std_image = stdCifar;
