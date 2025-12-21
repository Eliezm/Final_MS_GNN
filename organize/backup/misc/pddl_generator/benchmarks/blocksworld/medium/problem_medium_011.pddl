(define (problem blocksworld-medium-11)
  (:domain blocksworld)
  (:objects b0 b1 b2 b3 b4 b5 b6)
  (:init
    (arm-empty) (on-table b1) (on b0 b1) (clear b0) (on-table b3) (on b2 b3) (clear b2) (on-table b4) (clear b4) (on-table b5) (clear b5) (on-table b6) (clear b6)
  )
  (:goal (and
    (on-table b0) (on b1 b0) (on-table b2) (on b3 b2) (on-table b4) (on b5 b4) (on-table b6)
  ))
)
