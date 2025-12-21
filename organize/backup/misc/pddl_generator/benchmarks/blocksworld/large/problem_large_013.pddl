(define (problem blocksworld-large-13)
  (:domain blocksworld)
  (:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11)
  (:init
    (on-table b0) (on-table b1) (on-table b2) (on-table b3) (on-table b4) (on-table b5) (on-table b6) (on-table b7) (on-table b8) (on-table b9) (on-table b10) (on-table b11) (arm-empty) (clear b0) (clear b1) (clear b2) (clear b3) (clear b4) (clear b5) (clear b6) (clear b7) (clear b8) (clear b9) (clear b10) (clear b11)
  )
  (:goal (and
    (on b0 b1) (on b1 b2) (on b2 b3) (on b3 b4) (on b5 b6) (on b6 b7) (on b7 b8) (on b8 b9) (on-table b9) (on-table b10) (on-table b11)
  ))
)
