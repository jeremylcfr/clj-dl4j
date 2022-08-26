(ns clj-dl4j.datasets
  (:require [clj-datavec.records.csv :as csv]
            [clj-datavec.records.jackson :as jackson])
  (:import [org.deeplearning4j.datasets.datavec RecordReaderDataSetIterator]
           [org.datavec.api.records.reader RecordReader]
           [org.nd4j.linalg.dataset DataSet]))

(defn- infer-type-from-options
  [{:keys [batch-size label-idx max-labels max-batches label-idx-from label-idx-to] :as options}]
  (if batch-size
    (cond (and label-idx max-labels)
            :classification
          (and label-idx-from label-idx-to)
            :regression
          :else
            (throw (Exception. (str "Dataset iterator - Options : " options " do not define any kind of dataset policy"))))
    (throw (Exception. "Dataset iterator - Batch size is mandatory"))))

(defn- classification-record-reader-dataset-iterator
  ^RecordReaderDataSetIterator
  [{:keys [batch-size label-idx max-labels max-batches] :or {max-batches -1}} ^RecordReader record-reader]
  (RecordReaderDataSetIterator. ^RecordReader record-reader ^int (int batch-size) ^int (int label-idx) ^int (int max-labels) ^int (int max-batches)))

(defn- regression-record-reader-dataset-iterator
  ^RecordReaderDataSetIterator
  [{:keys [batch-size label-idx-from label-idx-to] :or {max-batches -1}} ^RecordReader record-reader]
  (RecordReaderDataSetIterator. ^RecordReader record-reader ^int (int batch-size) ^int (int label-idx-from) ^int (int label-idx-to) true))

(defn record-reader-dataset-iterator
  ^RecordReaderDataSetIterator
  ([options record-reader]
   (record-reader-dataset-iterator nil options record-reader))
  ([dataset-type options record-reader]
    (let [dataset-type (if dataset-type dataset-type (infer-type-from-options options))]
      (case dataset-type
            :classification (classification-record-reader-dataset-iterator options record-reader)
            :regression (regression-record-reader-dataset-iterator options record-reader)))))


(def formats
  {:csv {:indexed? true
         :reader-fn csv/->csv-record-reader
         :initializer csv/initialize!}
   :jackson {:indexed? false
             :reader-fn jackson/jackson-line-record-reader
             :initializer jackson/initialize!}})

(def conversions
  {:json    :jackson
   :jsons   :jackson
   :ndjson  :jackson
   :xml     :jackson
   :yaml    :jackson})


(defn index-options
  [positions {:keys [label-idx label-idx-to label-idx-from] :as options}]
  (cond-> options
          (keyword? label-idx)       (assoc :label-idx (get positions label-idx))
          (keyword? label-idx-from)  (assoc :label-idx-from (get positions label-idx-from))
          (keyword? label-idx-to)    (assoc :label-idx-to (get positions label-idx-to))))

(defn ->indexed-record-reader-dataset-iterator
  [{:keys [reader-fn initializer]} {:keys [use-nd4j? dataset-type] :as options} io-coercible]
  (let [record-reader (reader-fn options)]
    (initializer use-nd4j? io-coercible)
    (record-reader-dataset-iterator dataset-type options record-reader)))

(defn ->unindexed-record-reader-dataset-iterator
  [{:keys [reader-fn initializer]} {:keys [use-nd4j? dataset-type] :as options} io-coercible]
  (let [{:keys [record-reader positions]} (reader-fn options)]
    (initializer record-reader use-nd4j? io-coercible)
    (record-reader-dataset-iterator dataset-type (index-options positions options) record-reader)))

(defn ->record-reader-dataset-iterator
  ;; Inference, if path indicates it
  ^RecordReaderDataSetIterator
  ([options io-coercible])
  ([input-type options io-coercible]
   (let [input-type (get conversions input-type input-type)
         {:keys [indexed? reader-fn] :as io-material} (get formats input-type)]
     (if reader-fn
       (if indexed?
         (->indexed-record-reader-dataset-iterator io-material options io-coercible)
         (->unindexed-record-reader-dataset-iterator io-material options io-coercible))
       (throw (Exception. (str "No record reader specification found for type : " input-type ", this might be unavailable in this library")))))))

(defn read-from-dataset-iterator
  [^RecordReaderDataSetIterator reader]
  (loop [out (transient [])]
    (if (.hasNext ^RecordReaderDataSetIterator reader)
      (recur (conj! out (.next ^RecordReaderDataSetIterator reader)))
      (persistent! out))))

(defn read-datasets
  ([options io-coercible])
  ([input-type options io-coercible]
   (read-from-dataset-iterator
     (->record-reader-dataset-iterator input-type options io-coercible))))
