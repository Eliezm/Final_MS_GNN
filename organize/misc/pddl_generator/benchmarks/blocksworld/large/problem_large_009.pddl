(define (problem blocksworld-large-9)
  (:domain blocksworld)
  (:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12)
  (:init
    (arm-empty) (on-table b1) (on b0 b1) (clear b0) (on-table b4) (on b2 b3) (on b3 b4) (clear b2) (on-table b6) (on b5 b6) (clear b5) (on-table b8) (on b7 b8) (clear b7) (on-table b9) (clear b9) (on-table b10) (clear b10) (on-table b11) (clear b11) (on-table b12) (clear b12)
  )
  (:goal (and
    (on-table b12) (on b0 b12) (on b1 b0) (on b2 b0) (on b3 b1) (on-table b4) (on-table b5) (on b6 b2) (on b7 b0) (on b8 b7) (on b9 b3) (on-table b10) (on-table b11)
  ))
)
