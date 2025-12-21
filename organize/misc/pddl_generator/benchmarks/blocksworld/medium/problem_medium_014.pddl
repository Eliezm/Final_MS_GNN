(define (problem blocksworld-medium-14)
  (:domain blocksworld)
  (:objects b0 b1 b2 b3 b4 b5 b6 b7 b8)
  (:init
    (on-table b0) (on-table b1) (on-table b2) (on-table b3) (on-table b4) (on-table b5) (on-table b6) (on-table b7) (on-table b8) (arm-empty) (clear b0) (clear b1) (clear b2) (clear b3) (clear b4) (clear b5) (clear b6) (clear b7) (clear b8)
  )
  (:goal (and
    (on b0 b1) (on b1 b2) (on b2 b3) (on b4 b5) (on b5 b6) (on b6 b7) (on-table b7) (on-table b8)
  ))
)
