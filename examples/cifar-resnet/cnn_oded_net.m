function net = cnn_oded_net(varargin)
opts.k = [64 128 256];
opts.Nclass=10;%number of classses (CIFAR-10 / CIFAR-100)
opts.colorSpace = 'rgb';
opts.usePad = 1;
opts.useQR = 1;
opts = vl_argparse(opts, varargin) ;

net = dagnn.DagNN; %network

%initial convolution
if strcmp(opts.colorSpace, 'gray')
    c = 1;
else
    c = 3;
end

leakyConst = 1e-2;
layers = {};
% color space transformation
layers{end+1} = {'addConvBlk',[1 1 10],{'order',{'conv','bn','relu'},'downsample',0,'isPad',1,'bias',false}};
layers{end+1} = {'addConvBlk',[1 1 c],{'order',{'conv','bn','relu'},'downsample',0,'isPad',1,'bias',false}};


% layers{end+1} = {'addConvBlk',[3 3 32],{'order',{'conv','bn','relu'},'downsample',0,'leak',leakyConst,'isPad',1,'bias',false}};
% layers{end+1} = {'pooling', [2 2], 2};

% add resnet blocks
for iK = 1:length(opts.k)
    layers{end+1} = {'skipStart'};
    layers{end+1} = {'addConvBlk',[3 3 opts.k(iK)],{'order',{'conv','bn','relu'},'downsample',0,'leak',leakyConst,'isPad',1,'bias',false}};
    layers{end+1} = {'addConvBlk',[3 3 opts.k(iK)],{'order',{'conv','bn','relu'},'downsample',0,'leak',leakyConst,'isPad',1,'bias',false}};
%     layers{end+1} = {'addConvBlk',[1 1 opts.k(iK)/2],{'order',{'conv','bn','relu'},'downsample',0,'leak',leakyConst,'isPad',1,'bias',false}};
    layers{end+1} = {'addConvBlk',[3 3 opts.k(iK)],{'order',{'conv','bn','relu'},'downsample',0,'leak',leakyConst,'isPad',1,'bias',false}};
    layers{end+1} = {'skipEnd',[1 1],{'downsample',0,'order',{'conv'},'isPad',1}};
    layers{end+1} = {'skipStart'};
    layers{end+1} = {'addConvBlk',[3 3 opts.k(iK)],{'order',{'conv','bn','relu'},'downsample',0,'leak',leakyConst,'isPad',1,'bias',false}};
    layers{end+1} = {'addConvBlk',[3 3 opts.k(iK)],{'order',{'conv','bn','relu'},'downsample',0,'leak',leakyConst,'isPad',1,'bias',false}};
    layers{end+1} = {'addConvBlk',[3 3 opts.k(iK)],{'order',{'conv','bn','relu'},'downsample',0,'leak',leakyConst,'isPad',1,'bias',false}};
    layers{end+1} = {'skipEnd',[1 1],{'downsample',0,'order',{'conv'},'isPad',1}};
    layers{end+1} = {'pooling', [2 2], 2};
end
% layers{end+1} = {'skipStart'};
% layers{end+1} = {'addConvBlk',[3 3 128],{'order',{'conv','bn','relu'},'downsample',0,'leak',leakyConst,'isPad',1,'bias',false}};
% layers{end+1} = {'addConvBlk',[1 1 64],{'order',{'conv','bn','relu'},'downsample',0,'leak',leakyConst,'isPad',1,'bias',false}};
% layers{end+1} = {'addConvBlk',[3 3 128],{'order',{'conv','bn','relu'},'downsample',0,'leak',leakyConst,'isPad',1,'bias',false}};
% layers{end+1} = {'skipEnd',1,{'downsample',0,'order',{'conv'},'isPad',1}};
% layers{end+1} = {'pooling', [2 2], 2};
% layers{end+1} = {'skipStart'};
% layers{end+1} = {'addConvBlk',[3 3 256],{'order',{'conv','bn','relu'},'downsample',0,'leak',leakyConst,'isPad',1,'bias',false}};
% layers{end+1} = {'addConvBlk',[1 1 128],{'order',{'conv','bn','relu'},'downsample',0,'leak',leakyConst,'isPad',1,'bias',false}};
% layers{end+1} = {'addConvBlk',[3 3 256],{'order',{'conv','bn','relu'},'downsample',0,'leak',leakyConst,'isPad',1,'bias',false}};
% layers{end+1} = {'skipEnd',1,{'downsample',0,'order',{'conv'},'isPad',1}};
% layers{end+1} = {'pooling', [2 2], 2};

layers{end} = {'pooling', [4 4], 1, 'avg'};

net = buildNet(layers,'inputSize',[32 32 3],'batchNormalization',true);

% allBlockTypes = cellfun( @(c) class(c), {net.layers.block}, 'UniformOutput', false);
% c = net.layers( find( strcmp(allBlockTypes,'dagnn.Conv') ,1,'last') ).block.size(end);
c = opts.k(end);

%% add losses

lossTypes = { % class, iou, bbox_regression
    {'class',100,'softmaxlog'}
    {'class',20,'softmaxlog'}
    };
    

derOutputs = {};

lastOutputName = net.layers(end).outputs;

for iL = 1:length(lossTypes)
    switch lower(lossTypes{iL}{1})
        case 'class'
            % Class loss
            if length(lossTypes{iL}) < 3
                lossTypes{iL}{3} = 'softmaxlog';
            end
            lossNumOutput = lossTypes{iL}{2};
            lossName = ['class',num2str(iL),'_',num2str(lossNumOutput)];
            net.addLayer(['conv_',lossName], dagnn.Conv('size', [1 1 c lossNumOutput], 'hasBias', true), lastOutputName, ['conv_',lossName,'_out'], {['convW_',lossName], ['convB',lossName]} );
            net.addLayer(['loss_',lossName], dagnn.Loss('loss', lossTypes{iL}{3}), {['conv_',lossName,'_out'],['label_',lossName]},{['err_',lossName]});
            net.addLayer(['lossV_',lossName], dagnn.Loss('loss', 'classerror'), {['conv_',lossName,'_out'],['label_',lossName]},{['errV_',lossName]});
            derOutputs{end+1} = ['err_',lossName];
            derOutputs{end+1} = 2;
            
        case 'iou'
            % IoU loss
            lossNumOutput = 1;
            lossName = ['IoU',num2str(iL)];
            net.addLayer(['conv_',lossName], dagnn.Conv('size', [1 1 c lossNumOutput], 'hasBias', true), lastOutputName, ['conv_',lossName,'_out'], {['convW_',lossName], ['convB',lossName]} );
            net.addLayer(['sigmoid_',lossName], dagnn.Sigmoid(), ['conv_',lossName,'_out'], ['sigmoid_',lossName,'_out']);
            net.addLayer(['pdist_',lossName], dagnn.PDist('p',2, 'aggregate', true), {['sigmoid_',lossName,'_out'],['label_',lossName]}, {['err_',lossName]});
            % net.addLayer(['pdist_',lossName], dagnn.PDist('p',2,'noRoot',false,'epsilon', 1e-6, 'aggregate', true), {['conv_',lossName,'_out'],['label_',lossName]}, {['err_',lossName]});
            derOutputs{end+1} = ['err_',lossName];
            derOutputs{end+1} = 2; % error weigth

        case 'bbox_regression'
            % Regression loss
            if length(lossTypes{iL}) < 2
                lossTypes{iL}{2} = 5;
            end
            lossName = ['bboxRegression',num2str(iL)];
            lossNumOutput = lossTypes{iL}{2};
            net.addLayer(['conv_',lossName], dagnn.Conv('size', [1 1 c lossNumOutput], 'hasBias', true), lastOutputName, ['conv_',lossName,'_out'], helper(opts, {['convW_',lossName], ['convB',lossName]}) );
            net.addLayer(['sigmoid_',lossName], dagnn.Sigmoid(), ['conv_',lossName,'_out'], ['sigmoid_',lossName,'_out']);
            net.addLayer(['scale_',lossName], dagnn.Scale('size',1), ['sigmoid_',lossName,'_out'], ['scale_',lossName,'_out'], {['scaleW_',lossName],['scaleB_',lossName]});
            
            % set scaling value for setting the sigmoid output to be between [-1, 1]
            p = net.getParamIndex(net.layers(net.getLayerIndex(['scale_',lossName])).params) ;
            [net.params(p).learningRate] = deal(0);
            params = {single(2),single(-1)};
            [net.params(p).value] = deal(params{:}) ;
            
            net.addLayer(['loss_',lossName], dagnn.PDist('p',2, 'aggregate', true), {['conv_',lossName,'_out'],['label_',lossName]},{['err_',lossName]});
            % net.addLayer(['loss_',lossName], dagnn.PDist('p',2,'noRoot',false,'epsilon', 1e-6, 'aggregate', true), {['conv_',lossName,'_out'],['label_',lossName]},{['err_',lossName]});
            derOutputs{end+1} = ['err_',lossName];
            derOutputs{end+1} = 1;

    end
end

   
net.initParams();

if opts.useQR
    blockTypes = cellfun(@class,{net.layers.block},'UniformOutput',false);
    convLayerInds = strcmp( blockTypes, 'dagnn.Conv');
    convParamInds = [net.layers(convLayerInds).paramIndexes];
    conv3x3Inds = cellfun( @(c) size(net.params(c).value,1) > 1 && size(net.params(c).value,2) > 1, num2cell(convParamInds));
    conv3x3Inds = convParamInds(conv3x3Inds);
    
    for iC = 1:length(conv3x3Inds)
        p = net.params(conv3x3Inds(iC)).value;
        pOut = zeros(size(p),'like',p);
        for iP = 1:size(p,4)*size(p,3)
            pMat = p(:,:,iP);
            [pMatQ,pMatR] = qr(pMat);
            pOut(:,:,iP) = pMatQ * norm(pMat);
        end
        net.params(conv3x3Inds(iC)).value = pOut;
    end
end



%Meta parameters
net.meta.inputSize = [32 32 c] ;
net.meta.trainOpts.learningRate = [0.01*ones(1,10) 0.1*ones(1,80) 0.01*ones(1,40) 0.001*ones(1,40)] ;
net.meta.trainOpts.weightDecay = 0.0001 ;
net.meta.trainOpts.batchSize = 100 ;
net.meta.trainOpts.momentum = 0.9 ;
net.meta.trainOpts.derOutputs = derOutputs;
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;
end
