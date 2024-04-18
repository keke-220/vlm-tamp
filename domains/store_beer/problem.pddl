(define (problem store_beer-0)
    (:domain omnigibson)

    (:objects
        beer_bottle-n-01_1 beer_bottle-n-01_2 beer_bottle-n-01_3 - beer_bottle-n-01
        electric_refrigerator-n-01_1 - electric_refrigerator-n-01
        floor-n-01_1 - floor-n-01
        agent-n-01_1 - agent-n-01
        kitchen - room
    )

    (:init
        (closed electric_refrigerator-n-01_1)
        (inroom beer_bottle-n-01_1 kitchen)
        (inroom beer_bottle-n-01_2 kitchen)
        (inroom beer_bottle-n-01_3 kitchen)
        (inroom electric_refrigerator-n-01_1 kitchen)
        (inroom floor-n-01_1 kitchen)
        (inroom agent-n-01_1 kitchen)
        (handempty agent-n-01_1)
    )

    (:goal
        (and
            (inside beer_bottle-n-01_1 electric_refrigerator-n-01_1)
            (inside beer_bottle-n-01_2 electric_refrigerator-n-01_1)
            (inside beer_bottle-n-01_3 electric_refrigerator-n-01_1)
        )
    )
)