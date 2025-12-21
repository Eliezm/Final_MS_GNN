(define (problem parking-medium-0)
  (:domain parking)
  (:objects car0 car1 car2 car3 - car curb0 curb1 curb2 curb3 curb4 - curb)
  (:init
    (at-curb-num car0 curb0) (car-clear car0) (at-curb-num car1 curb1) (car-clear car1) (at-curb-num car2 curb2) (car-clear car2) (at-curb-num car3 curb3) (car-clear car3) (curb-clear curb4) (at-curb car0)
  )
  (:goal (and
    (at-curb-num car1 curb0) (at-curb-num car0 curb1) (at-curb-num car2 curb2) (at-curb-num car3 curb3)
  ))
)
