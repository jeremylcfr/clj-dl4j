(ns clj-dl4j.layers.supertypes.layer
  (:require [clj-dl4j.dropout :as dropout]
            [clj-dl4j.constraints :as constraints])
  (:import [org.deeplearning4j.nn.conf.layers Layer Layer$Builder]
           [org.deeplearning4j.nn.conf.dropout IDropout]
           [org.deeplearning4j.nn.api.layers LayerConstraint]))

(defn add-constraints-from-map
  ^Layer$Builder
  [^Layer$Builder builder {:keys [all weights bias]}]
  (cond-> builder
          (and all (not (empty? all)))             (.constrainAllParameters ^Layer$Builder #^"[Lorg.deeplearning4j.nn.api.layers.LayerConstraint;" (constraints/->constraints all))
          (and weights (not (empty? weights)))     (.constrainWeights ^Layer$Builder #^"[Lorg.deeplearning4j.nn.api.layers.LayerConstraint;" (constraints/->constraints weights))
          (and bias (not (empty? bias)))           (.constrainBias ^Layer$Builder #^"[Lorg.deeplearning4j.nn.api.layers.LayerConstraint;" (constraints/->constraints bias))))

(defn add-constraints-from-seq
  ^Layer$Builder
 [^Layer$Builder builder constraints]
  (add-constraints-from-map builder
    {:all (filter (fn [{:keys [scope]}] (or (nil? scope) (= :all scope))) constraints)
     :weights (filter (fn [{:keys [scope]}] (= :weights scope)) constraints)
     :bias (filter (fn [{:keys [scope]}] (= :bias scope)) constraints)}))

(defn add-constraints
  ^Layer$Builder
  [^Layer$Builder builder conf]
  (if (map? conf)
    (add-constraints-from-map builder conf)
    (add-constraints-from-seq builder conf)))

(defn build-with
  ^Layer$Builder
  [{:keys [name dropout constraints]} ^Layer$Builder builder]
  (cond-> builder
          name         (.name ^Layer$Builder ^String (str name))
          dropout      (.dropOut ^Layer$Builder ^IDropout (dropout/->dropout dropout))
          constraints  (add-constraints constraints)))
