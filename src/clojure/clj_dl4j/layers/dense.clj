(ns clj-dl4j.layers.dense
  (:require [clj-dl4j.layers.supertypes.feedforward :as super])
  (:import [org.deeplearning4j.nn.conf.layers DenseLayer DenseLayer$Builder]))

(defn dense-layer-builder
  ^DenseLayer$Builder
  ([]
   (DenseLayer$Builder.))
  ([{:keys [has-bias?] :as options}]
   (let [builder (dense-layer-builder)]
     (cond-> (super/build-with options builder)
             (boolean? has-bias?)          (.hasBias ^DenseLayer$Builder has-bias?)))))

(defn dense-layer-builder?
  [obj]
  (instance? DenseLayer$Builder obj))

(defn ->dense-layer-builder
  ^DenseLayer$Builder
  [obj]
  (if (dense-layer-builder? obj)
    obj
    (dense-layer-builder obj)))

(defn dense-layer
  ^DenseLayer
  [options]
  (.build ^DenseLayer$Builder (->dense-layer-builder options)))

(defn dense-layer?
  [obj]
  (instance? DenseLayer obj))

(defn ->dense-layer
  ^DenseLayer
  [obj]
  (if (dense-layer? obj)
    obj
    (dense-layer obj)))
