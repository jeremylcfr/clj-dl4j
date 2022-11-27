(defproject io.github.jeremylcfr/clj-dl4j "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Apache License"
            :url "https://www.apache.org/licenses/LICENSE-2.0"}
  :dependencies [[org.clojure/clojure "1.11.1"]
                 [org.clojure/algo.generic "0.1.3"]
                 [io.github.jeremylcfr/clj-nd4j "0.1.0-SNAPSHOT"]
                 [io.github.jeremylcfr/clj-datavec "0.1.0-SNAPSHOT"]
                 [org.deeplearning4j/deeplearning4j-core "1.0.0-M2.1"]]
  :scm {:name "git" :url "https://github.com/jeremylcfr/clj-dl4j"}
  :source-paths ["src/clojure"]
  :java-source-paths ["src/java"]
  :profiles {:dev {:source-paths ["repl"]}}
  :repl-options {:init-ns repl})
