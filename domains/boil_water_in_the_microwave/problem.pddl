(define (problem boil_water_in_the_microwave-0)
    (:domain omnigibson)

    (:objects
        water-n-06_1 - water-n-06
        ; cooked__water-n-01_1 - cooked__water-n-01
        mug-n-04_1 - mug-n-04
        cabinet-n-01_1 - cabinet-n-01
        sink-n-01_1 - sink-n-01
        floor-n-01_1 - floor-n-01
        microwave-n-02_1 - microwave-n-02
        agent-n-01_1 - agent-n-01
        kitchen - room
    )

    (:init
        (closed cabinet-n-01_1)
        (closed microwave-n-02_1)
        (not (turnedon microwave-n-02_1))
        (not (cooked water-n-06_1))
        (inside mug-n-04_1 cabinet-n-01_1)
        (insource sink-n-01_1 water-n-06_1)
        (inroom floor-n-01_1 kitchen)
        (inroom sink-n-01_1 kitchen)
        (inroom microwave-n-02_1 kitchen)
        (inroom cabinet-n-01_1 kitchen)
        ; (ontop agent-n-01_1 floor-n-01_1)
        (handempty agent-n-01_1)
        ; (future cooked__water-n-01_1)
        (inroom agent-n-01_1 kitchen)
    )

    (:goal
        (and
            ;;(real ?cooked__water-n-01_1)
            ;;(filled ?mug-n-04_1 ?cooked__water-n-01_1) 
            ; (inside mug-n-04_1 microwave-n-02_1)
            ; (inview agent-n-01_1 cabinet-n-01_1)
            ; (open cabinet-n-01_1)
            ; (inhand agent-n-01_1 mug-n-04_1)
            ; (filled mug-n-04_1 water-n-06_1)
            ;(inside mug-n-04_1 microwave-n-02_1)
            (cooked water-n-06_1)
            ;(turnedon microwave-n-02_1)
        )
    )
)