(ns clj-dl4j.layers.lstm
  (:require [clj-dl4j.layers.supertypes.lstm :as super])
  (:import [org.deeplearning4j.nn.conf.layers GravesLSTM GravesLSTM$Builder]))

(defn graves-lstm-builder
  ^GravesLSTM$Builder
  ([]
   (GravesLSTM$Builder.))
  ([options]
   (let [builder (graves-lstm-builder)]
     (super/build-with options builder))))

(defn graves-lstm-builder?
  [obj]
  (instance? GravesLSTM$Builder obj))

(defn ->graves-lstm-builder
  ^GravesLSTM$Builder
  [obj]
  (if (graves-lstm-builder? obj)
    obj
    (graves-lstm-builder obj)))

(defn graves-lstm
  ^GravesLSTM
  [options]
  (.build ^GravesLSTM$Builder (->graves-lstm-builder options)))

(defn graves-lstm?
  [obj]
  (instance? GravesLSTM obj))

(defn ->graves-lstm
  ^GravesLSTM
  [obj]
  (if (graves-lstm? obj)
    obj
    (graves-lstm obj)))
  
   