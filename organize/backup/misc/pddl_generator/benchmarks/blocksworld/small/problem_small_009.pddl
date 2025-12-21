(define (problem blocksworld-small-9)
  (:domain blocksworld)
  (:objects b0 b1 b2 b3)
  (:init
    (arm-empty) (on-table b1) (on b0 b1) (clear b0) (on-table b2) (clear b2) (on-table b3) (clear b3)
  )
  (:goal (and
    (on-table b3) (on b0 b3) (on b1 b0) (on b2 b1)
  ))
)
