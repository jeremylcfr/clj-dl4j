(ns clj-dl4j.convolution
  (:import [org.deeplearning4j.nn.conf ConvolutionMode]
           [org.deeplearning4j.nn.conf.layers Convolution3D Convolution3D$DataFormat]
           [org.deeplearning4j.nn.conf.layers Layer ConvolutionLayer ConvolutionLayer$AlgoMode]))

(defn ->convolution-mode
  ^ConvolutionMode
  [mode]
  (case mode
        :strict    ConvolutionMode/Strict
        :truncate  ConvolutionMode/Truncate
        :same      ConvolutionMode/Same))

(defn ->convolution-data-format
  ^Convolution3D$DataFormat
  [fmt]
  (case fmt
        :ncdhw Convolution3D$DataFormat/NCDHW
        :ndhwc Convolution3D$DataFormat/NDHWC))

(defn ->cuda-convolution-mode
  ^ConvolutionLayer$AlgoMode
  [mode]
  (case mode
        :no-workspace ConvolutionLayer$AlgoMode/NO_WORKSPACE
        :fastest ConvolutionLayer$AlgoMode/PREFER_FASTEST
        (throw (Exception. (str "CUDA - Cuda convolution mode : " mode " not found, should be :no-workspace or :fastest, user-specified not supported for now")))))


