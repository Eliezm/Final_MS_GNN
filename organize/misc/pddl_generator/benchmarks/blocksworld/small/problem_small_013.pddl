(define (problem blocksworld-small-13)
  (:domain blocksworld)
  (:objects b0 b1 b2 b3)
  (:init
    (on-table b0) (on-table b1) (on-table b2) (on-table b3) (arm-empty) (clear b0) (clear b1) (clear b2) (clear b3)
  )
  (:goal (and
    (on b0 b1) (on-table b2) (on-table b3)
  ))
)
