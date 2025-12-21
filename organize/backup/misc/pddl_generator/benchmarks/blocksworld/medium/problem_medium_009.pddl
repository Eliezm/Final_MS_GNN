(define (problem blocksworld-medium-9)
  (:domain blocksworld)
  (:objects b0 b1 b2 b3 b4 b5 b6 b7)
  (:init
    (arm-empty) (on-table b1) (on b0 b1) (clear b0) (on-table b4) (on b2 b3) (on b3 b4) (clear b2) (on-table b5) (clear b5) (on-table b6) (clear b6) (on-table b7) (clear b7)
  )
  (:goal (and
    (on-table b7) (on b0 b7) (on b1 b0) (on b2 b0) (on b3 b1) (on-table b4) (on-table b5) (on b6 b2)
  ))
)
