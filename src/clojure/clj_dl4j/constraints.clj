(ns clj-dl4j.constraints
  (:require [clj-dl4j.utils :refer [->ints-args ->string-set]])
  (:import [org.deeplearning4j.nn.api.layers LayerConstraint]
           [org.deeplearning4j.nn.conf.constraint BaseConstraint MaxNormConstraint MinMaxNormConstraint NonNegativeConstraint UnitNormConstraint]))

;;////////////////////////////////////////////////////////////////////////////
;;============================================================================
;;                                   GENERIC
;;============================================================================
;;\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

;;=====================================================
;;=======================BUILDER=======================
;;=====================================================

;;=================Raw===============

;; Adapt dimensions from layers types situationaly...if possible
;; More for convenience

(defn max-norm-constraint
  ^MaxNormConstraint
  [{:keys [max-norm dimensions param-names]}]
  (if param-names
    (MaxNormConstraint. ^double (double max-norm) (->string-set param-names) ^ints (->ints-args dimensions))
    (MaxNormConstraint. ^double (double max-norm) ^ints (->ints-args dimensions))))

;; Maybe add validation if really relevant...
(defn min-max-norm-constraint
  ^MinMaxNormConstraint
  [{:keys [min-norm max-norm rate dimensions param-names]}]
  (cond param-names
          (MinMaxNormConstraint. ^double (double min-norm) ^double (double max-norm) ^double (double rate) (->string-set param-names) ^ints (->ints-args dimensions))
        rate
          (MinMaxNormConstraint. ^double (double min-norm) ^double (double max-norm) ^double (double rate) ^ints (->ints-args dimensions))
        :else
          (MinMaxNormConstraint. ^double (double min-norm) ^double (double max-norm) ^ints (->ints-args dimensions))))

(defn non-negative-constraint
  ^NonNegativeConstraint
  [options]
  (NonNegativeConstraint.))

(defn unit-norm-constraint
  ^UnitNormConstraint
  [{:keys [dimensions param-names]}]
  (if param-names
    (UnitNormConstraint. (->string-set param-names) ^ints (->ints-args dimensions))
    (UnitNormConstraint. ^ints (->ints-args dimensions))))

(def raw-builders
  {:max-norm max-norm-constraint
   :min-max-norm min-max-norm-constraint
   :non-negative non-negative-constraint
   :unit-norm unit-norm-constraint})

(defn constraint
  ^LayerConstraint
  [{:keys [type] :as options}]
  (if-let [builder (get raw-builders type)]
    (builder options)
    (throw (Exception. (str "LAYER CONSTRAINT - Unknown constraint type : " type)))))


(defn constraint?
  [obj]
  (instance? LayerConstraint obj))

(defn ->constraint
  ^LayerConstraint
  [obj]
  (if (constraint? obj)
    obj
    (constraint obj)))

(defn ->constraints
  #^"[Lorg.deeplearning4j.nn.api.layers.LayerConstraint;"
  [obj]
  (into-array LayerConstraint
    (if (or (sequential? obj) (set? obj))
      (map ->constraint obj)
      [(->constraint obj)])))



