begin_version
3
end_version
begin_metric
0
end_metric
2
begin_variable
var0
-1
2
Atom robotattarget(rob, targ)
NegatedAtom robotattarget(rob, targ)
end_variable
begin_variable
var1
-1
2
Atom taskcomplete(rob, targ)
NegatedAtom taskcomplete(rob, targ)
end_variable
0
begin_state
1
1
end_state
begin_goal
1
1 0
end_goal
2
begin_operator
move_to_loc_same_room rob r0 targ
0
1
0 0 -1 0
1
end_operator
begin_operator
sleep rob targ
1
0 0
1
0 1 -1 0
1
end_operator
0
