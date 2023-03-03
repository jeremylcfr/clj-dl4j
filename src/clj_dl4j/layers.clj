(ns clj-dl4j.layers
  (:require [clj-dl4j.layers [convolution :as convolution]
                             [dense :as dense]
                             [dropout :as dropout]
                             [loss :as loss]
                             [lstm :as lstm]
                             [output :as output]]
            [clj-dl4j.layers.lstm :as lstm])
  (:import [org.deeplearning4j.nn.conf.layers Layer]))

(def layer-fns
  {:convolution                 convolution/->convolution-layer
   :convolution-1d              convolution/->convolution-1d-layer
   :convolution-3d              convolution/->convolution-3d-layer
   :deconvolution-2d            convolution/->deconvolution-2d-layer
   :depth-wise-convolution-2d   convolution/->depth-wise-convolution-2d-layer
   :dense                       dense/->dense-layer
   :dropout                     dropout/->dropout-layer
   :loss                        loss/->loss-layer
   :graves-lstm                 lstm/->graves-lstm
   :output                      output/->output-layer
   :rnn-loss                    loss/->rnn-loss-layer
   :rnn-output                  output/->rnn-output-layer})

(defn layer
  ^Layer
  [{:keys [type] :as options}]
  (if-let [layer-fn (get layer-fns type)]
    (layer-fn options)
    (throw (Exception. (str "LAYER - Unknown layer type : " type)))))

(defn layer?
  [obj]
  (instance? Layer obj))

(defn ->layer
  ^Layer
  [obj]
  (if (layer? obj)
    layer
    (layer obj)))
