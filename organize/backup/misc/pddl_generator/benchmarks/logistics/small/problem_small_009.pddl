(define (problem logistics-small-9)
  (:domain logistics)
  (:objects
    truck-city0 truck-city1 - truck
    loc-city0-0 loc-city0-1 loc-city0-2 loc-city1-0 loc-city1-1 loc-city1-2 - location
    pkg0 pkg1 pkg2 - object
    city0 city1 - city
  )
  (:init
    (at-obj pkg0 loc-city0-1) (obj-at-city pkg0 city0) (at-obj pkg1 loc-city0-1) (obj-at-city pkg1 city0) (at-obj pkg2 loc-city0-1) (obj-at-city pkg2 city0) (at truck-city0 loc-city0-0) (truck-at-city truck-city0 city0) (at truck-city1 loc-city1-0) (truck-at-city truck-city1 city1) (connected loc-city0-0 loc-city0-1) (connected loc-city0-1 loc-city0-0) (connected loc-city0-1 loc-city0-2) (connected loc-city0-2 loc-city0-1) (connected loc-city1-0 loc-city1-1) (connected loc-city1-1 loc-city1-0) (connected loc-city1-1 loc-city1-2) (connected loc-city1-2 loc-city1-1) (connected loc-city0-0 loc-city1-0) (connected loc-city1-0 loc-city0-0)
  )
  (:goal (and
    (at-obj pkg0 loc-city0-2) (at-obj pkg1 loc-city0-0) (at-obj pkg2 loc-city0-1)
  ))
)
