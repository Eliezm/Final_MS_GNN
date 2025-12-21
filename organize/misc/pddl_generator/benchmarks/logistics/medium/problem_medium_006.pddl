(define (problem logistics-medium-6)
  (:domain logistics)
  (:objects
    truck-city0 truck-city1 truck-city2 - truck
    loc-city0-0 loc-city0-1 loc-city1-0 loc-city1-1 loc-city2-0 loc-city2-1 - location
    pkg0 pkg1 pkg2 pkg3 pkg4 pkg5 - object
    city0 city1 city2 - city
  )
  (:init
    (at-obj pkg0 loc-city1-0) (obj-at-city pkg0 city1) (at-obj pkg1 loc-city0-0) (obj-at-city pkg1 city0) (at-obj pkg2 loc-city1-0) (obj-at-city pkg2 city1) (at-obj pkg3 loc-city0-0) (obj-at-city pkg3 city0) (at-obj pkg4 loc-city2-0) (obj-at-city pkg4 city2) (at-obj pkg5 loc-city2-0) (obj-at-city pkg5 city2) (at truck-city0 loc-city0-0) (truck-at-city truck-city0 city0) (at truck-city1 loc-city1-0) (truck-at-city truck-city1 city1) (at truck-city2 loc-city2-0) (truck-at-city truck-city2 city2) (connected loc-city0-0 loc-city0-1) (connected loc-city0-1 loc-city0-0) (connected loc-city1-0 loc-city1-1) (connected loc-city1-1 loc-city1-0) (connected loc-city2-0 loc-city2-1) (connected loc-city2-1 loc-city2-0) (connected loc-city0-0 loc-city1-0) (connected loc-city1-0 loc-city0-0) (connected loc-city1-0 loc-city2-0) (connected loc-city2-0 loc-city1-0)
  )
  (:goal (and
    (at-obj pkg0 loc-city0-0) (at-obj pkg1 loc-city1-0) (at-obj pkg2 loc-city1-1) (at-obj pkg3 loc-city1-0) (at-obj pkg4 loc-city2-0) (at-obj pkg5 loc-city0-0)
  ))
)
