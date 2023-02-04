(ns clj-dl4j.layers.supertypes.recurrent
  (:require [clj-dl4j.layers.supertypes.feedforward :as super])
  (:import [org.deeplearning4j.nn.conf.layers BaseRecurrentLayer BaseRecurrentLayer$Builder]))


;; Add later
(defn build-with
  ^BaseRecurrentLayer$Builder
  [{:keys [n-in units n-out] :as options} ^BaseRecurrentLayer$Builder builder]
  (super/build-with options builder))