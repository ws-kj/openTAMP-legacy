; AUTOGENERATED. DO NOT EDIT.

(define (domain robotics)                  
(:requirements :strips :equality :typing)
(:types
Robot Room Wall Location Target DomainSettings - base_object)

(:predicates
(RobotInRoom ?var0 - Robot ?var1 - Room )
(LocationInRoom ?var0 - Location ?var1 - Room )
(TaskComplete ?var0 - Robot ?var1 - Target )
(IsMP ?var0 - Robot )
)

(:action move_to_loc_same_room
:parameters (?robot - Robot ?room - Room ?target - Target)  
:precondition (and (RobotInRoom ?robot ?room)) 
:effect (and (RobotInRoom ?robot ?room)(IsMP ?robot)(TaskComplete ?robot ?target))
)

)