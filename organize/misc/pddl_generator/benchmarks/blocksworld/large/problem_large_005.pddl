(define (problem blocksworld-large-5)
  (:domain blocksworld)
  (:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12)
  (:init
    (arm-empty) (on-table b2) (on b0 b1) (on b1 b2) (clear b0) (on-table b5) (on b3 b4) (on b4 b5) (clear b3) (on-table b8) (on b6 b7) (on b7 b8) (clear b6) (on-table b9) (clear b9) (on-table b10) (clear b10) (on-table b11) (clear b11) (on-table b12) (clear b12)
  )
  (:goal (and
    (on-table b2) (on b0 b1) (on b1 b2) (on-table b5) (on b3 b4) (on b4 b5) (on-table b8) (on b6 b7) (on b7 b8) (on-table b12) (on b9 b10) (on b10 b11) (on b11 b12)
  ))
)
