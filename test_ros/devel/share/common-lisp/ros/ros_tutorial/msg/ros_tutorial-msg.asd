
(cl:in-package :asdf)

(defsystem "ros_tutorial-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :actionlib_msgs-msg
               :std_msgs-msg
)
  :components ((:file "_package")
    (:file "counterAction" :depends-on ("_package_counterAction"))
    (:file "_package_counterAction" :depends-on ("_package"))
    (:file "counterActionFeedback" :depends-on ("_package_counterActionFeedback"))
    (:file "_package_counterActionFeedback" :depends-on ("_package"))
    (:file "counterActionGoal" :depends-on ("_package_counterActionGoal"))
    (:file "_package_counterActionGoal" :depends-on ("_package"))
    (:file "counterActionResult" :depends-on ("_package_counterActionResult"))
    (:file "_package_counterActionResult" :depends-on ("_package"))
    (:file "counterFeedback" :depends-on ("_package_counterFeedback"))
    (:file "_package_counterFeedback" :depends-on ("_package"))
    (:file "counterGoal" :depends-on ("_package_counterGoal"))
    (:file "_package_counterGoal" :depends-on ("_package"))
    (:file "counterResult" :depends-on ("_package_counterResult"))
    (:file "_package_counterResult" :depends-on ("_package"))
  ))