(ns clj-dl4j.layers.lstm
  (:require [clj-dl4j.layers.supertypes.lstm :as super])
  (:import [org.deeplearning4j.nn.conf.layers LSTM LSTM$Builder GravesLSTM GravesLSTM$Builder]))

;; Too much redundancy
;; Build is for super, refactor

(defn lstm-builder
  ^LSTM$Builder
  ([]
   (LSTM$Builder.))
  ([options]
   (let [builder (lstm-builder)]
     (super/build-with options builder))))

(defn lstm-builder?
  [obj]
  (instance? LSTM$Builder obj))

(defn ->lstm-builder
  ^LSTM$Builder
  [obj]
  (if (lstm-builder? obj)
    obj
    (lstm-builder obj)))

(defn lstm
  ^LSTM
  [options]
  (.build ^LSTM$Builder (->lstm-builder options)))

(defn lstm?
  [obj]
  (instance? LSTM obj))

(defn ->lstm
  ^LSTM
  [obj]
  (if (lstm? obj)
    obj
    (lstm obj)))



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
  
   