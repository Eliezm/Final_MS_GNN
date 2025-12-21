(define (problem logistics-large-5)
  (:domain logistics)
  (:objects
    truck-city0 truck-city1 truck-city2 truck-city3 - truck
    loc-city0-0 loc-city0-1 loc-city0-2 loc-city1-0 loc-city1-1 loc-city1-2 loc-city2-0 loc-city2-1 loc-city2-2 loc-city3-0 loc-city3-1 loc-city3-2 - location
    pkg0 pkg1 pkg2 pkg3 pkg4 pkg5 pkg6 pkg7 pkg8 pkg9 pkg10 pkg11 pkg12 pkg13 pkg14 - object
    city0 city1 city2 city3 - city
  )
  (:init
    (at-obj pkg0 loc-city1-0) (obj-at-city pkg0 city1) (at-obj pkg1 loc-city0-2) (obj-at-city pkg1 city0) (at-obj pkg2 loc-city2-2) (obj-at-city pkg2 city2) (at-obj pkg3 loc-city1-1) (obj-at-city pkg3 city1) (at-obj pkg4 loc-city3-2) (obj-at-city pkg4 city3) (at-obj pkg5 loc-city2-2) (obj-at-city pkg5 city2) (at-obj pkg6 loc-city1-2) (obj-at-city pkg6 city1) (at-obj pkg7 loc-city1-1) (obj-at-city pkg7 city1) (at-obj pkg8 loc-city2-1) (obj-at-city pkg8 city2) (at-obj pkg9 loc-city1-0) (obj-at-city pkg9 city1) (at-obj pkg10 loc-city0-0) (obj-at-city pkg10 city0) (at-obj pkg11 loc-city1-0) (obj-at-city pkg11 city1) (at-obj pkg12 loc-city3-1) (obj-at-city pkg12 city3) (at-obj pkg13 loc-city3-2) (obj-at-city pkg13 city3) (at-obj pkg14 loc-city0-2) (obj-at-city pkg14 city0) (at truck-city0 loc-city0-0) (truck-at-city truck-city0 city0) (at truck-city1 loc-city1-0) (truck-at-city truck-city1 city1) (at truck-city2 loc-city2-0) (truck-at-city truck-city2 city2) (at truck-city3 loc-city3-0) (truck-at-city truck-city3 city3) (connected loc-city0-0 loc-city0-1) (connected loc-city0-1 loc-city0-0) (connected loc-city0-1 loc-city0-2) (connected loc-city0-2 loc-city0-1) (connected loc-city1-0 loc-city1-1) (connected loc-city1-1 loc-city1-0) (connected loc-city1-1 loc-city1-2) (connected loc-city1-2 loc-city1-1) (connected loc-city2-0 loc-city2-1) (connected loc-city2-1 loc-city2-0) (connected loc-city2-1 loc-city2-2) (connected loc-city2-2 loc-city2-1) (connected loc-city3-0 loc-city3-1) (connected loc-city3-1 loc-city3-0) (connected loc-city3-1 loc-city3-2) (connected loc-city3-2 loc-city3-1) (connected loc-city0-0 loc-city1-0) (connected loc-city1-0 loc-city0-0) (connected loc-city1-0 loc-city2-0) (connected loc-city2-0 loc-city1-0) (connected loc-city2-0 loc-city3-0) (connected loc-city3-0 loc-city2-0)
  )
  (:goal (and
    (at-obj pkg0 loc-city2-2) (at-obj pkg1 loc-city2-2) (at-obj pkg2 loc-city0-1) (at-obj pkg3 loc-city0-2) (at-obj pkg4 loc-city0-0) (at-obj pkg5 loc-city3-1) (at-obj pkg6 loc-city2-1) (at-obj pkg7 loc-city1-1) (at-obj pkg8 loc-city2-0) (at-obj pkg9 loc-city3-1) (at-obj pkg10 loc-city3-1) (at-obj pkg11 loc-city2-0) (at-obj pkg12 loc-city3-0) (at-obj pkg13 loc-city0-0) (at-obj pkg14 loc-city3-0)
  ))
)
