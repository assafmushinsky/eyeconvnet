classdef Sigmoid2 < dagnn.ElementWise
  properties
    scale = 1
    offset = 0
  end
  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnsigmoid2(inputs{1},[],obj.scale,obj.offset) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = vl_nnsigmoid2(inputs{1}, derOutputs{1},obj.scale,obj.offset) ;
      derParams = {} ;
    end
    
    function obj = Sigmoid2(varargin)
      obj.load(varargin) ;
    end
  end
end
