begin_version
3
end_version
begin_metric
0
end_metric
5
begin_variable
var0
-1
2
Atom certainposition(targ)
NegatedAtom certainposition(targ)
end_variable
begin_variable
var1
-1
2
Atom facedloc(rob)
NegatedAtom facedloc(rob)
end_variable
begin_variable
var2
-1
2
Atom movedtoloc(rob)
NegatedAtom movedtoloc(rob)
end_variable
begin_variable
var3
-1
2
Atom robotattarget(rob, targ)
NegatedAtom robotattarget(rob, targ)
end_variable
begin_variable
var4
-1
2
Atom taskcomplete(rob)
NegatedAtom taskcomplete(rob)
end_variable
0
begin_state
1
1
1
1
1
end_state
begin_goal
1
4 0
end_goal
5
begin_operator
approach_targ rob targ r0
1
0 0
2
0 2 -1 1
0 3 -1 0
1
end_operator
begin_operator
approach_vantage rob targ r0 vantage
1
1 0
1
0 2 -1 0
1
end_operator
begin_operator
move_near_loc_and_observe rob targ r0
0
1
0 1 -1 0
1
end_operator
begin_operator
view_vantage rob targ r0
0
3
0 0 -1 0
0 1 -1 1
0 2 0 1
1
end_operator
begin_operator
zzzzz rob targ
1
3 0
1
0 4 -1 0
1
end_operator
0
