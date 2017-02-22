function net = buildNet(layers,varargin)
%BUILDNET build a CNN structure according to cell array LAYERS
%
% NET = BUILDNET(LAYERS,...) start new DagNN and add layers according to
% specified in cell array LAYERS. Each cell in LAYERS correspond to a layer
% in the final NET. Each cell contain option cell array of option for the new
% layer. The first option cell contain a string to specify the layer type
% and the rest specifies option for this layer.
%
% For example, LAYERS{l} contains the options for layer number l.
% LAYERS{l}{1} contain a string specifing the layer type and
% LAYERS{l}{2:end} contains options for this specific layer.
%
% BUILDNET(..., 'Option', value) takes the following options:
%
%   `inputSize`:: [height width output_channels] (default: [32 32 3])
%       Specifies the input image size.
%
%   `batchNormalization`:: true or false (default: true)
%       Whether to add batch normalization for the convRelu layer type.
%       This option can be turned off in the middle of the net.
%
% Possible layer types and thier options:
%   'convRelu': add conv layer with optional batch normalization and relu.
%       LAYERS{l}{2} - conv size: [height width output_channels].
%       LAYERS{l}{3} - conv stride: one or two elements vector.
%       LAYERS{l}{4} - boolean to add batch normalization (if set in options) and relu (default: true).
% 
%   'pooling': add pooling layer.
%       LAYERS{l}{2} - pooling size: [height width].
%       LAYERS{l}{3} - pooling stride: one or two elements vector.
%       LAYERS{l}{4} - pooling method: max, avg (default: max).
%
%   'dropout': add dropout layer.
%       LAYERS{l}{2} - dropout rate.
%       
%   'BNoff': turn off batch normalization option to the next layers.
%
%   'addConvBlk': add any combination of conv, batch normalization and relu layers. 
%       LAYERS{l}{2} - conv size: [height width output_channels].
%       LAYERS{l}{3} - cell array of addConvBlk options {'Option',value,...}:
%                      'order' - set the exist and order of the layers 
%                                using cell array of the following strings: 'bn', 'relu', 'conv'
%                                (default: {'conv','bn','relu'})
%                      'downsample' - boolean for setting the stride to 1 or 2.
%                      'leak' - if bigger than 0, use leaky relu with the set parameter.
%                      'isPad' - boolean whether to pad the conv layer.
%                      'bais' - boolean whether to add bais to the conv layer.
%   
%   'skipStart': save location for skip connection
%   'skipEnd': connect the location saved in skipStart to the currect
%              location using conv and sum layers.
%       LAYERS{l}{2} - conv size: [height width].
%       LAYERS{l}{3} - cell array of addConvBlk options {'Option',value,...}, see above:
%


opts.inputSize = [32 32 3];
opts.batchNormalization = 0;

opts = vl_argparse(opts,varargin);

helper = @(o,c) c(1:2-o.batchNormalization);

net = dagnn.DagNN();

c = opts.inputSize(3);
inputVarName = 'input';
addAfter.var = inputVarName;
addAfter.depth = c;
skipStart = addAfter;
for l = 1:length(layers)
    if length(layers{l}) < 3, layers{l}{3} = 1; end
    switch layers{l}{1}
        case 'convRelu'
            if length(layers{l}) > 3
                doBnRelu = layers{l}{4};
            else
                doBnRelu = true;
            end
            
            net.addLayer(['conv',num2str(l)], dagnn.Conv('size', [layers{l}{2}(1:2), c, layers{l}{2}(3)], 'stride', [1 1].*layers{l}{3}, 'hasBias', ~opts.batchNormalization), inputVarName, [], helper(opts, {['convW',num2str(l)], ['convB',num2str(l)]}));
            c = layers{l}{2}(3);
            if doBnRelu
                if opts.batchNormalization, net.addLayer(['bnorm',num2str(l)], dagnn.BatchNorm('numChannels',c), [], [], {['bnormW',num2str(l)],['bnormB',num2str(l)], ['bnormMuSigma',num2str(l)]} ); end;
                net.addLayer(['relu',num2str(l)], dagnn.ReLU());
            end

        case 'pooling'
            method = 'max';
            if length(layers{l}) > 3
                method = layers{l}{4};
            end
            net.addLayer(['pool',num2str(l)], dagnn.Pooling('poolSize', layers{l}{2}(1:2), 'stride', [1 1].*layers{l}{3}, 'pad', [0 0 0 0], 'method', method));
            
        case 'dropout'
            net.addLayer(['dropout',num2str(l)], dagnn.DropOut('rate', layers{l}{2}));
        case 'BNoff'
            opts.batchNormalization = 0;
        case 'crop'
%             dagnn.Crop
        case 'resBlk33'
            addAfter.var = net.layers(end).outputs{end};
            addAfter.depth = c;
            resStart = addAfter;
            
            [net,addAfter] = AddConvBlock(net,addAfter,['resA',num2str(l)],layers{l}{2}(1:2),layers{l}{2}(3),...
                'downsample',layers{l}{3}>1,'order',{'bn','relu','conv'});
            [net,addAfter] = AddConvBlock(net,addAfter,['resB',num2str(l)],layers{l}{2}(1:2),layers{l}{2}(3),...
                'downsample',false,'order',{'bn','relu','conv'});
            [net,addAfterCut] = AddConvBlock(net,resStart,['resCut',num2str(l)],[1 1],layers{l}{2}(3),...
                'downsample',layers{l}{3}>1,'order',{'conv'});
            
            net.addLayer(['resSum',num2str(l)], dagnn.Sum(), {addAfterCut.var,addAfter.var}, ['sumOut',num2str(l)]);
            
            c = addAfter.depth;
        case 'addConvBlk'
            if l == 1
                addAfter.var = inputVarName;
                addAfter.depth = c;
            else
                addAfter.var = net.layers(end).outputs{end};
                addAfter.depth = c;
            end
            [net,addAfter] = AddConvBlock(net,addAfter,['convBlk',num2str(l)],layers{l}{2}(1:2),layers{l}{2}(3),...
                layers{l}{3}{:});
            c = addAfter.depth;
        case 'skipStart'
            addAfter.var = net.layers(end).outputs{end};
            addAfter.depth = c;
            skipStart = addAfter;
        case 'skipEnd'
            
            [net,addAfterCut] = AddConvBlock(net,skipStart,['skipCut',num2str(l)],layers{l}{2},addAfter.depth,...
                layers{l}{3}{:});
            
            net.addLayer(['resSum',num2str(l)], dagnn.Sum(), {addAfterCut.var,addAfter.var}, ['sumOut',num2str(l)]);
            
            c = addAfter.depth;
    end
    inputVarName = [];
    
    addAfter.var = net.layers(end).outputs{end};
    addAfter.depth = c;
end
