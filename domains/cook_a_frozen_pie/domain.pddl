(define (domain omnigibson)

    (:requirements :strips :typing :negative-preconditions :conditional-effects)

    (:types
        movable liquid furniture room agent - object

        mug pie - movable
        water - liquid
        electric_refrigerator oven cabinet sink floor microwave - furniture

        kitchen - room

        water-n-06 - water
        mug-n-04 - mug
        cabinet-n-01 - cabinet
        sink-n-01 - sink
        floor-n-01 - floor
        microwave-n-02 - microwave
        pie-n-01 - pie
        oven-n-01 - oven
        electric_refrigerator-n-01 - electric_refrigerator

        agent-n-01 - agent
    )

    (:predicates
        (inside ?o1 - object ?o2 - object)
        (insource ?s - sink ?w - liquid)
        (inroom ?o - object ?r - room)
        (inhand ?a - agent ?o - object)
        (inview ?a - agent ?o - object)
        (handempty ?a - agent)
        (closed ?o - object)
        (filled ?o - object ?w - liquid)
        (turnedon ?o - object)
        (cooked ?o - object)
        (found ?a - agent ?o - object)
        (frozen ?o - object)
        (hot ?o - object)
    )

    (:action find
        :parameters (?a - agent ?o - object ?r - room)
        :precondition (and (inroom ?a ?r) (inroom ?o ?r))
        :effect (and (inview ?a ?o) (found ?a ?o) (forall
                (?oo - object)
                (when
                    (found ?a ?oo)
                    (not (found ?a ?oo)))))
    )

    (:action grasp
        :parameters (?a - agent ?o1 - movable ?o2 - object)
        :precondition (and (inview ?a ?o1) (found ?a ?o1) (handempty ?a))
        :effect (and (not (inview ?a ?o1)) (not (handempty ?a)) (inhand ?a ?o1))
    )

    (:action placein
        :parameters (?a - agent ?o1 - movable ?o2 - object)
        :precondition (and (not (handempty ?a)) (inhand ?a ?o1) (inview ?a ?o2) (found ?a ?o2) (not (closed ?o2)))
        :effect (and (handempty ?a) (not (inhand ?a ?o1)) (inside ?o1 ?o2))
    )

    (:action fillsink
        :parameters (?a - agent ?s - sink ?w - liquid)
        :precondition (and (inview ?a ?s) (found ?a ?s) (insource ?s ?w) (not (filled ?s ?w)))
        :effect (filled ?s ?w)
    )

    (:action fill
        :parameters (?a - agent ?o - movable ?s - sink ?w - liquid)
        :precondition (and (inhand ?a ?o) (filled ?s ?w) (not (filled ?o ?w)) (inview ?a ?s) (found ?a ?s))
        :effect (and (filled ?o ?w) (not (filled ?s ?w)))
    )

    (:action openit
        :parameters (?a - agent ?o - object ?r - room)
        :precondition (and (inview ?a ?o) (found ?a ?o) (inroom ?o ?r))
        :effect (and (not (closed ?o)) (forall
                (?oo - object)
                (when
                    (inside ?oo ?o)
                    (inroom ?oo ?r))
            ))
    )

    (:action closeit
        :parameters (?a - agent ?o - object ?r - room)
        :precondition (and (inview ?a ?o) (found ?a ?o) (inroom ?o ?r))
        :effect (and (closed ?o) (forall
                (?oo - object)
                (when
                    (inside ?oo ?o)
                    (not (inroom ?oo ?r)))
                    ; (and (not (inroom ?oo ?r)) (not (inview ?a ?oo))))
            ))
    )

    (:action microwave_water
        :parameters (?a - agent ?m - microwave ?o - movable ?w - water)
        :precondition (and (inview ?a ?m) (found ?a ?m) (closed ?m) (inside ?o ?m) (filled ?o ?w)) ; this will potentially cause problem, where the robot will stuck at checking if the microwave has the cup inside
        :effect (and (turnedon ?m) (cooked ?w))
    )

    (:action heat_food_with_oven
        :parameters (?a - agent ?v - oven ?f - object)
        :precondition (and (inview ?a ?v) (found ?a ?v) (inside ?f ?v))
        :effect (and (hot ?f) (turnedon ?v))
    )
)