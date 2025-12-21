(define (problem parking-small-10)
  (:domain parking)
  (:objects car0 car1 car2 - car curb0 curb1 curb2 curb3 - curb)
  (:init
    (at-curb-num car0 curb0) (car-clear car0) (at-curb-num car1 curb1) (car-clear car1) (at-curb-num car2 curb2) (car-clear car2) (curb-clear curb3) (at-curb car0)
  )
  (:goal (and
    (at-curb-num car0 curb0) (at-curb-num car1 curb1) (at-curb-num car2 curb2)
  ))
)
