(define (problem blocksworld-small-6)
  (:domain blocksworld)
  (:objects b0 b1 b2 b3)
  (:init
    (arm-empty) (on-table b0) (clear b0) (on-table b1) (clear b1) (on-table b2) (clear b2) (on-table b3) (clear b3)
  )
  (:goal (and
    (on-table b1) (on b0 b1) (on-table b3) (on b2 b3)
  ))
)
