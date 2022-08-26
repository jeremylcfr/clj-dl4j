(ns clj-dl4j.layers.output
  (:require [clj-dl4j.layers.supertypes.base-output :as super])
  (:import [org.deeplearning4j.nn.conf.layers OutputLayer OutputLayer$Builder RnnOutputLayer RnnOutputLayer$Builder]))

(defn output-layer-builder
  ^OutputLayer$Builder
  ([]
   (OutputLayer$Builder.))
  ([options]
   (let [builder (output-layer-builder)]
     (super/build-with options builder))))

(defn output-layer-builder?
  [obj]
  (instance? OutputLayer$Builder obj))

(defn ->output-layer-builder
  ^OutputLayer$Builder
  [obj]
  (if (output-layer-builder? obj)
    obj
    (output-layer-builder obj)))

(defn output-layer
  [options]
  (.build ^OutputLayer$Builder (->output-layer-builder options)))

(defn output-layer?
  [obj]
  (instance? OutputLayer obj))

(defn ->output-layer
  ^OutputLayer
  [obj]
  (if (output-layer? obj)
    obj
    (output-layer obj)))

(defn rnn-output-layer-builder
  ^RnnOutputLayer$Builder
  ([]
   (RnnOutputLayer$Builder.))
  ([options]
   (let [builder (rnn-output-layer-builder)]
     (super/build-with options builder))))

(defn rnn-output-layer-builder?
  [obj]
  (instance? RnnOutputLayer$Builder obj))

(defn ->rnn-output-layer-builder
  ^RnnOutputLayer$Builder
  [obj]
  (if (rnn-output-layer-builder? obj)
    obj
    (rnn-output-layer-builder obj)))

(defn rnn-output-layer
  [options]
  (.build ^RnnOutputLayer$Builder (->rnn-output-layer-builder options)))

(defn rnn-output-layer?
  [obj]
  (instance? RnnOutputLayer obj))

(defn ->rnn-output-layer
  ^RnnOutputLayer
  [obj]
  (if (rnn-output-layer? obj)
    obj
    (rnn-output-layer obj)))

