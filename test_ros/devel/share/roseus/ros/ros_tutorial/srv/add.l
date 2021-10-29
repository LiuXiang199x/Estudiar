;; Auto-generated. Do not edit!


(when (boundp 'ros_tutorial::add)
  (if (not (find-package "ROS_TUTORIAL"))
    (make-package "ROS_TUTORIAL"))
  (shadow 'add (find-package "ROS_TUTORIAL")))
(unless (find-package "ROS_TUTORIAL::ADD")
  (make-package "ROS_TUTORIAL::ADD"))
(unless (find-package "ROS_TUTORIAL::ADDREQUEST")
  (make-package "ROS_TUTORIAL::ADDREQUEST"))
(unless (find-package "ROS_TUTORIAL::ADDRESPONSE")
  (make-package "ROS_TUTORIAL::ADDRESPONSE"))

(in-package "ROS")





(defclass ros_tutorial::addRequest
  :super ros::object
  :slots (_A _B ))

(defmethod ros_tutorial::addRequest
  (:init
   (&key
    ((:A __A) 0)
    ((:B __B) 0)
    )
   (send-super :init)
   (setq _A (round __A))
   (setq _B (round __B))
   self)
  (:A
   (&optional __A)
   (if __A (setq _A __A)) _A)
  (:B
   (&optional __B)
   (if __B (setq _B __B)) _B)
  (:serialization-length
   ()
   (+
    ;; int32 _A
    4
    ;; int32 _B
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; int32 _A
       (write-long _A s)
     ;; int32 _B
       (write-long _B s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; int32 _A
     (setq _A (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; int32 _B
     (setq _B (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;;
   self)
  )

(defclass ros_tutorial::addResponse
  :super ros::object
  :slots (_sum ))

(defmethod ros_tutorial::addResponse
  (:init
   (&key
    ((:sum __sum) 0)
    )
   (send-super :init)
   (setq _sum (round __sum))
   self)
  (:sum
   (&optional __sum)
   (if __sum (setq _sum __sum)) _sum)
  (:serialization-length
   ()
   (+
    ;; int32 _sum
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; int32 _sum
       (write-long _sum s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; int32 _sum
     (setq _sum (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;;
   self)
  )

(defclass ros_tutorial::add
  :super ros::object
  :slots ())

(setf (get ros_tutorial::add :md5sum-) "13918c933de1bcb345f89e4d03db37f4")
(setf (get ros_tutorial::add :datatype-) "ros_tutorial/add")
(setf (get ros_tutorial::add :request) ros_tutorial::addRequest)
(setf (get ros_tutorial::add :response) ros_tutorial::addResponse)

(defmethod ros_tutorial::addRequest
  (:response () (instance ros_tutorial::addResponse :init)))

(setf (get ros_tutorial::addRequest :md5sum-) "13918c933de1bcb345f89e4d03db37f4")
(setf (get ros_tutorial::addRequest :datatype-) "ros_tutorial/addRequest")
(setf (get ros_tutorial::addRequest :definition-)
      "int32 A
int32 B
---
int32 sum

")

(setf (get ros_tutorial::addResponse :md5sum-) "13918c933de1bcb345f89e4d03db37f4")
(setf (get ros_tutorial::addResponse :datatype-) "ros_tutorial/addResponse")
(setf (get ros_tutorial::addResponse :definition-)
      "int32 A
int32 B
---
int32 sum

")



(provide :ros_tutorial/add "13918c933de1bcb345f89e4d03db37f4")


