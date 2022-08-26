(ns clj-dl4j.log
  (:import [org.deeplearning4j.optimize.api TrainingListener]
           [org.deeplearning4j.optimize.listeners ScoreIterationListener]))


(defn score-iteration-listener
  ^ScoreIterationListener
  [{:keys [nb-iterations] :or {nb-iterations 10}}]
  (ScoreIterationListener. ^int (int nb-iterations)))

(defn score-iteration-listener?
  [obj]
  (instance? ScoreIterationListener obj))

(defn ->score-iteration-listener
  ^ScoreIterationListener
  [obj]
  (if (score-iteration-listener? obj)
    obj
    (score-iteration-listener obj)))

(def training-listeners
  {:score-iteration     ->score-iteration-listener})

(defn training-listener?
  [obj]
  (instance? TrainingListener obj))

(defn ->training-listener
  ^TrainingListener
  [obj]
  (if (training-listener? obj)
    obj
    (if-let [builder-fn (get training-listeners (:type obj))]
      (builder-fn obj)
      (throw (Exception. (str "TRAINING LISTENER - Unknow listener type : " (:type obj)))))))





