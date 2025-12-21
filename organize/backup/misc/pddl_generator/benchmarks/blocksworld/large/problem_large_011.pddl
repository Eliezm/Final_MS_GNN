(define (problem blocksworld-large-11)
  (:domain blocksworld)
  (:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11)
  (:init
    (arm-empty) (on-table b1) (on b0 b1) (clear b0) (on-table b3) (on b2 b3) (clear b2) (on-table b5) (on b4 b5) (clear b4) (on-table b7) (on b6 b7) (clear b6) (on-table b8) (clear b8) (on-table b9) (clear b9) (on-table b10) (clear b10) (on-table b11) (clear b11)
  )
  (:goal (and
    (on-table b0) (on b1 b0) (on-table b2) (on b3 b2) (on-table b4) (on b5 b4) (on-table b6) (on b7 b6) (on-table b8) (on b9 b8) (on-table b10) (on b11 b10)
  ))
)
