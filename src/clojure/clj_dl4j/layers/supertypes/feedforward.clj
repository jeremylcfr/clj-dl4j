(ns clj-dl4j.layers.supertypes.feedforward
  (:require [clj-dl4j.layers.supertypes.base :as super])
  (:import [org.deeplearning4j.nn.conf.layers FeedForwardLayer FeedForwardLayer$Builder]))

(defn build-with
  ^FeedForwardLayer$Builder
  [{:keys [n-in units n-out] :as options} ^FeedForwardLayer$Builder builder]
  (when (and units n-out)
    (if (= units n-out)
      (println "[Warning] LAYER - Beware, units and n-out are aliases, does not throw since they are equal")
      (throw (Exception. (str "LAYER - Ambiguous specification, n-out = " n-out " and units= " units " depspite the second aliasing the first. Choose one")))))
  (cond-> (super/build-with options builder)
          n-in     (.nIn  ^FeedForwardLayer$Builder ^int (int n-in))
          n-out    (.nOut ^FeedForwardLayer$Builder ^int (int n-out))
          units    (.nOut ^FeedForwardLayer$Builder ^int (int units))))
