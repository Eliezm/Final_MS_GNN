(define (problem blocksworld-medium-10)
  (:domain blocksworld)
  (:objects b0 b1 b2 b3 b4 b5 b6)
  (:init
    (arm-empty) (on-table b2) (on b0 b1) (on b1 b2) (clear b0) (on-table b5) (on b3 b4) (on b4 b5) (clear b3) (on-table b6) (clear b6)
  )
  (:goal (and
    (on-table b6) (on b0 b6) (on b1 b0) (on b2 b0) (on b3 b0) (on-table b4) (on-table b5)
  ))
)
