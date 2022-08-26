(ns clj-dl4j.layers.loss
  (:require [clj-dl4j.layers.supertypes.feedforward :as super]
            [clj-dl4j.layers.supertypes.base-output :as rnnsuper])
  (:import [org.deeplearning4j.nn.conf.layers LossLayer LossLayer$Builder RnnLossLayer RnnLossLayer$Builder]))

(defn loss-layer-builder
  ^LossLayer$Builder
  ([]
   (LossLayer$Builder.))
  ([{:keys [has-bias?] :as options}]
   (let [builder (loss-layer-builder)]
     (cond-> (super/build-with options builder)
             (boolean? has-bias?)          (.hasBias ^LossLayer$Builder has-bias?)))))

(defn loss-layer-builder?
  [obj]
  (instance? LossLayer$Builder obj))

(defn ->loss-layer-builder
  ^LossLayer$Builder
  [obj]
  (if (loss-layer-builder? obj)
    obj
    (loss-layer-builder obj)))

(defn loss-layer
  ^LossLayer
  [options]
  (.build ^LossLayer$Builder (->loss-layer-builder options)))

(defn loss-layer?
  [obj]
  (instance? LossLayer obj))

(defn ->loss-layer
  ^LossLayer
  [obj]
  (if (loss-layer? obj)
    obj
    (loss-layer obj)))

(defn rnn-loss-layer-builder
  ^RnnLossLayer$Builder
  ([]
   (RnnLossLayer$Builder.))
  ([{:keys [has-bias?] :as options}]
   (let [builder (rnn-loss-layer-builder)]
     (rnnsuper/build-with options builder))))

(defn rnn-loss-layer-builder?
  [obj]
  (instance? RnnLossLayer$Builder obj))

(defn ->rnn-loss-layer-builder
  ^RnnLossLayer$Builder
  [obj]
  (if (rnn-loss-layer-builder? obj)
    obj
    (rnn-loss-layer-builder obj)))

(defn rnn-loss-layer
  ^RnnLossLayer
  [options]
  (.build ^RnnLossLayer$Builder (->rnn-loss-layer-builder options)))

(defn rnn-loss-layer?
  [obj]
  (instance? RnnLossLayer obj))

(defn ->rnn-loss-layer
  ^RnnLossLayer
  [obj]
  (if (rnn-loss-layer? obj)
    obj
    (rnn-loss-layer obj)))
