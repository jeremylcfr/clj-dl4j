(ns clj-dl4j.utils
  (:require [clj-java-commons.core :refer [->int-array]]))

(defn ->ints-args
  ^ints
  [obj]
  (if (number? obj)
    (->int-array [obj])
    (->int-array obj)))

(defn ->string-set
  [coll]
  (persistent!
    (reduce
      (fn [agg x]
        (conj! agg (cond (string? x)
                           x
                         (keyword? x)
                          (name x)
                         :else
                           x)))
      (transient #{}) coll)))

