(define (problem store_brownies-0)
    (:domain omnigibson)

    (:objects
        brownie-n-03_1 brownie-n-03_2 - brownie-n-03
        electric_refrigerator-n-01_1 - electric_refrigerator-n-01
        tupperware-n-01_1 - tupperware-n-01
        floor-n-01_1 - floor-n-01
        agent-n-01_1 - agent-n-01
        kitchen - room
    )

    (:init
        (closed electric_refrigerator-n-01_1)
        (inroom brownie-n-03_1 kitchen)
        (inroom brownie-n-03_2 kitchen)
        (inroom tupperware-n-01_1 kitchen)
        (inroom electric_refrigerator-n-01_1 kitchen)
        (inroom floor-n-01_1 kitchen)
        (inroom agent-n-01_1 kitchen)
        (handempty agent-n-01_1)
    )

    (:goal
        (and
            (inside brownie-n-03_1 tupperware-n-01_1)
            (inside brownie-n-03_2 tupperware-n-01_1)
            ; (inside brownie-n-03_3 tupperware-n-01_1)
            (inside tupperware-n-01_1 electric_refrigerator-n-01_1)
        )
    )
)