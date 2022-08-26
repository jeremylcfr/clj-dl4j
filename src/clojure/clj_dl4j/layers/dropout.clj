(ns clj-dl4j.layers.dropout
  (:require [clj-dl4j.layers.supertypes.feedforward :as super])
  (:import [org.deeplearning4j.nn.conf.layers DropoutLayer DropoutLayer$Builder]))

(defn dropout-layer-builder
  ^DropoutLayer$Builder
  ([]
   (DropoutLayer$Builder.))
  ([options]
   (let [builder (dropout-layer-builder)]
     (super/build-with options builder))))

(defn dropout-layer-builder?
  [obj]
  (instance? DropoutLayer$Builder obj))

(defn ->dropout-layer-builder
  ^DropoutLayer$Builder
  [obj]
  (if (dropout-layer-builder? obj)
    obj
    (dropout-layer-builder obj)))

(defn dropout-layer
  ^DropoutLayer
  [options]
  (.build ^DropoutLayer$Builder (->dropout-layer-builder options)))

(defn dropout-layer?
  [obj]
  (instance? DropoutLayer obj))

(defn ->dropout-layer
  ^DropoutLayer
  [obj]
  (if (dropout-layer? obj)
    obj
    (dropout-layer obj)))
