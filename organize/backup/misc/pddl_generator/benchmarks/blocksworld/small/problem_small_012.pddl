(define (problem blocksworld-small-12)
  (:domain blocksworld)
  (:objects b0 b1 b2 b3 b4)
  (:init
    (arm-empty) (on-table b0) (clear b0) (on-table b1) (clear b1) (on-table b2) (clear b2) (on-table b3) (clear b3) (on-table b4) (clear b4)
  )
  (:goal (and
    (on-table b0) (on b1 b0) (on-table b2) (on b3 b2) (on-table b4)
  ))
)
