(define (problem blocksworld-medium-5)
  (:domain blocksworld)
  (:objects b0 b1 b2 b3 b4 b5 b6 b7)
  (:init
    (arm-empty) (on-table b2) (on b0 b1) (on b1 b2) (clear b0) (on-table b5) (on b3 b4) (on b4 b5) (clear b3) (on-table b6) (clear b6) (on-table b7) (clear b7)
  )
  (:goal (and
    (on-table b1) (on b0 b1) (on-table b3) (on b2 b3) (on-table b7) (on b4 b5) (on b5 b6) (on b6 b7)
  ))
)
