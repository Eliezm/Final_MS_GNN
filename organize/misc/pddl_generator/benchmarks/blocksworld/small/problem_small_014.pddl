(define (problem blocksworld-small-14)
  (:domain blocksworld)
  (:objects b0 b1 b2 b3 b4)
  (:init
    (on-table b0) (on-table b1) (on-table b2) (on-table b3) (on-table b4) (arm-empty) (clear b0) (clear b1) (clear b2) (clear b3) (clear b4)
  )
  (:goal (and
    (on b0 b1) (on b2 b3) (on-table b3) (on-table b4)
  ))
)
