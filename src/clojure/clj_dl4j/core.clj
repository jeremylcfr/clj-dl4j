(ns clj-dl4j.core
  (:require [clj-dl4j.configuration :as config]
            [clj-dl4j.log :as log])
  (:import [org.deeplearning4j.nn.multilayer MultiLayerNetwork]
           [org.deeplearning4j.nn.conf MultiLayerConfiguration]
           [org.deeplearning4j.optimize.listeners ScoreIterationListener]))

;; Alter to fit the tower
(defn multi-layer-network
  ^MultiLayerNetwork
  [{:keys [training-listener] :as options}]
  (let [configuration (config/network-configuration options)
        network (MultiLayerNetwork. ^MultiLayerConfiguration configuration)]
    (.init ^MultiLayerNetwork network)
    ;; Make it flexible, accept collection or single arg -> collection
    (when training-listener
      (.setListeners ^MultiLayerNetwork network [(log/->training-listener training-listener)]))
    network))

