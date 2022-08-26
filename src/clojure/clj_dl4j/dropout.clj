(ns clj-dl4j.dropout
  (:import [org.deeplearning4j.nn.conf.dropout IDropout Dropout AlphaDropout GaussianDropout GaussianNoise SpatialDropout]))


;; Schedule if needed

(defn standard-dropout
  ^Dropout
  [probability]
  (Dropout. ^double (double probability)))

(defn alpha-dropout
  ^AlphaDropout
  [probability]
  (AlphaDropout. ^double (double probability)))

(defn gaussian-dropout
  ^GaussianDropout
  [probability]
  (GaussianDropout. ^double (double probability)))

(defn gaussian-noise
  ^GaussianNoise
  [probability]
  (GaussianNoise. ^double (double probability)))

(defn spatial-dropout
  ^SpatialDropout
  [probability]
  (SpatialDropout. ^double (double probability)))

(def dropouts
  {:standard standard-dropout
   :alpha alpha-dropout
   :gaussian gaussian-dropout
   :gaussian-noise gaussian-noise
   :spatial spatial-dropout})

(defn dropout
  ^IDropout
  ([{:keys [type probability]}]
   (dropout type probability))
  ([type-fn probability]
   (if-let [builder (get dropouts type-fn)]
     (builder probability)
     (throw (Exception. (str "DROPOUT - Unknown dropout type : " type-fn))))))


(defn dropout?
  [obj]
  (instance? IDropout obj))

(defn ->dropout
  ^IDropout
  [obj]
  (if (dropout? obj)
    obj
    (if (map? obj)
      (dropout obj)
      (dropout :standard obj))))


