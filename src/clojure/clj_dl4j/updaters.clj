(ns clj-dl4j.updaters
  (:require [clj-nd4j.ml.gradient :as impl])
  (:import [org.deeplearning4j.nn.conf Updater]
           [org.nd4j.linalg.learning.config IUpdater]))

(def updaters
  {:sgd Updater/SGD
   :adam Updater/ADAM
   :adamax Updater/ADAMAX
   :adadelta Updater/ADADELTA
   :nesterovs Updater/NESTEROVS
   :nadam Updater/NADAM
   :adagrad Updater/ADAGRAD
   :rmsprop Updater/RMSPROP
   :none Updater/NONE})

(defn updater?
  [obj]
  (instance? IUpdater obj))

(defn ->updater
  ^IUpdater
  [obj]
  (if (updater? obj)
    obj
    (if (map? obj)
      (impl/->gradient-updater-config obj)
      (if-let [impl (get updaters (if obj obj :none))]
        (.getIUpdaterWithDefaultConfig ^Updater impl)
        (throw (Exception. (str "UPDATER - Unknown updater type : " obj)))))))
