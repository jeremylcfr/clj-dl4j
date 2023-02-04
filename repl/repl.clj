(ns repl
  (:require [clj-nd4j.ndarray :as nda]
            [clj-datavec.records [csv :as nscv]
                                 [jackson :as njack]]
            [clj-dl4j.core :as dl4j]
            [clj-dl4j.examples [xor :as xor]
                               [mnistanomaly :as mnist]
                               [regression :as reg]]
            [clj-java-commons.core :refer :all]
            [clj-java-commons.coerce :refer [->clj]])
  (:refer-clojure :exclude [/]))