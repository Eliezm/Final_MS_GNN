(define (problem blocksworld-large-10)
  (:domain blocksworld)
  (:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11)
  (:init
    (arm-empty) (on-table b2) (on b0 b1) (on b1 b2) (clear b0) (on-table b5) (on b3 b4) (on b4 b5) (clear b3) (on-table b8) (on b6 b7) (on b7 b8) (clear b6) (on-table b9) (clear b9) (on-table b10) (clear b10) (on-table b11) (clear b11)
  )
  (:goal (and
    (on-table b11) (on b0 b11) (on-table b1) (on b2 b0) (on b3 b0) (on-table b4) (on-table b5) (on b6 b0) (on b7 b2) (on-table b8) (on b9 b2) (on b10 b0)
  ))
)
