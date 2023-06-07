#!/usr/bin/env wolframscript
(* ::Package:: *)

(*A prewritten package*)
Get["/home/fb/Dokumentumok/Wolfram Mathematica/myRubik/Classes.m"]


(*The package written by me*)
Get["/home/fb/Dokumentumok/Wolfram Mathematica/myRubik/Rubikfunctions.m"]



(*
Motivation: group theory

COLOURS:
0 WHITE
1 YELLOW
2 BLUE
3 GREEN
4 ORANGE
5 RED

FACES:
F: FRONT
B: BACK
U: UP
D: DOWN (BOTTOM)
R: RIGHT
L: LEFT

CUBE:
Every facet of the cubies is marked with a number.
Central elements are not marked, because their relative position always remain the same.
All together that means 6*8=48 numbers, represented by a one dimensional list.

A solved cube with proper colours:
       3 3 3               
       3 U 3
       3 3 3
0 0 0  4 4 4  1 1 1  5 5 5
0 F 0  4 R 4  1 B 1  5 L 5
0 0 0  4 4 4  1 1 1  5 5 5
       2 2 2 
       2 D 2
       2 2 2
      
The positions of the faces in the list: 
       25 26 27
       28  U 29
       30 31 32
1 2 3  33 34 35  9  10 11  41 42 43
4 F 5  36  R 37  12  B 13  44  L 45
6 7 8  38 39 40  14 15 16  46 47 48
       17 18 19 
       20  D 21
       22 23 24 

*)


(*The Rubik cube class:
	- every face of the cubies has a number marking its colour
	- the whole cube is represented with a one dimensional list, called "cbs"
	- "name" can label the cube with a string
	- has 4 attributes: 
		1. cubies - gives back the actual state of the cube
		2. horizontal - turns the cube around a horizontal axis downward
		3. vertical - turns the cube around a vertical axis clockwise
		4. face - rotate the front face of the cube counterclockwise
With these operators, every permutation is possible.
*)
ClearAll[Rubik]
Class[ Rubik, Object,
       {cbs, name},
       {
        {new,      (new[super]; cbs = #1; name = #2)&},
        {cubies,  cbs&},
        {horizontal, Function[cbs=horizontalTurn[cbs]]},
        {vertical, Function[cbs=verticalTurn[cbs]]},
        {face, Function[cbs=faceTurn[cbs]]},
        {label,    name&}
       }
];



(*Create a solved cube and try the operators on it.*)
k={};Do[k=Join[k,ConstantArray[i,8]],{i,0,5}];
a=new[Rubik, k, "Solved Cube"]
cubies[a]
label[a]
horizontal[a]
vertical[a]

(*If the order of the operators changes, it gives different result.*)
k={};Do[k=Join[k,ConstantArray[i,8]],{i,0,5}];
a2=new[Rubik, k, "Solved Cube v2"]
cubies[a2]
label[a2]
vertical[a2]
horizontal[a2]


(*If the front face is rotated 4 times, it gives the identical operator.*)
label[a]

cubies[a]
Do[face[a],4]
cubies[a]

(*1 single rotation changes the colours.*)
label[a]
face[a]


(*These are the possible face rotations without getting different orientations in the end.
They make the rotations more visible and traceable for a human.*)
front[r_,number_Integer:1]:=(Do[face[r],number])
back[r_,number_Integer:1]:=(Do[vertical[r],2];Do[face[r],number];Do[vertical[r],2])
right[r_,number_Integer:1]:=(Do[vertical[r],1];Do[face[r],number];Do[vertical[r],3])
left[r_,number_Integer:1]:=(Do[vertical[r],3];Do[face[r],number];Do[vertical[r],1])
top[r_,number_Integer:1]:=(Do[horizontal[r],1];Do[face[r],number];Do[horizontal[r],3])
down[r_,number_Integer:1]:=(Do[horizontal[r],3];Do[face[r],number];Do[horizontal[r],1])

label[a2]

(*Rotating the down face counterclockwise.*)
cubies[a2]
down[a2]
cubies[a2]

(*Rotating the down face 3 more times, the original cube is obtained.*)
down[a2,3]
cubies[a2]


(*Create a mixed cube representation ( - a real cube was used to help reading the colours in the proper order).*)
l={4,3,4,5,5,2,2,5,2,1,1,3,3,0,1,5,0,1,4,0,4,5,0,3,1,4,2,4,2,0,3,5,3,5,1,2,1,3,5,2,4,2,3,0,0,1,4,0};
b=new[Rubik, l, "Mixed Cube"];
label[b]
cubies[b]


(*To solve the cube, a very basic algorithm was chosen, which is easy to learn for a human.
It starts with getting a cross on the top layer. The white colour was assigned to solve the cube for, marked with number 0.
To do it, the position of the Edge Cubies were defined in the following way as a list of lists:

edgecubies={{{UP,FRONT},{UP,RIGHT},{UP,BACK},{UP,LEFT}},
				{{FRONT,RIGHT},{RIGHT,BACK},{BACK,LEFT},{LEFT,FRONT}},
					{{DOWN,FRONT},{DOWN,RIGHT},{DOWN,BACK},{DOWN,LEFT}}}
					
In the next steps, only the side faces and the top face will rotate.
First, all 4 edge cubies should be transformed to the top layer.
*)

upperEdges[r_]:=Module[{
	sides={front,right,back,left},(*list of the same type of rotations*)
	edgecubies={{{28,2},{31,34},{29,10},{26,42}},{{5,36},{37,12},{13,44},{45,4}},{{20,7},{18,39},{21,15},{23,47}}}
	(*the positions of the edge cubies*)
						},
	For[i=1,i<4,i=Length[Extract[Cases[Position[edgecubies/.{x_,y_}:>{cubies[r][[x]],cubies[r][[y]]},0],{1,_,_}],{All,2}]],
	(*While all 4 edge cubies are not in the top layer, the following operations act:*)
			
		pos=Position[edgecubies/.{x_,y_}:>{cubies[r][[x]],cubies[r][[y]]},0];(*zero means, the cube is going to be solved for the "white" colour*)
		(*the positions of all edge cubies replaced by their colours *)
		middleEdgeNumber=Extract[Cases[pos,{2,_,_}],{All,2}];(*the positions of edge cubies in the middle layer, if any*)
		bottomEdgeNumber=Extract[Cases[pos,{3,_,_}],{All,2}];(*the positions of edge cubies in the bottom layer, if any*)
		Switch[middleEdgeNumber,
			(*Case1: they is at least 1 white cubie in the middle layer*)
			{__},If[MemberQ[#,middleEdgeNumber[[1]]]&@Extract[Cases[pos,{1,_,_}],{All,2}],
					(*check if there is already a white cubie above it, if yes, turn the top face before turning the side*)
					top[r,Mod[Complement[{1,2,3,4},Extract[Cases[pos,{1,_,_}],{All,2}]][[1]]+4-i,4]];sides[[middleEdgeNumber[[1]]]]@r,
					sides[[middleEdgeNumber[[1]]]]@r
					],
			(*Case2: if there are no white cubies in the middle layer*)
			{},  If[MemberQ[#,bottomEdgeNumber[[1]]]&@Extract[Cases[pos,{1,_,_}],{All,2}],
					(*the same actions except that the side face is turning twice in this case*)
					top[r,Mod[Complement[{1,2,3,4},Extract[Cases[pos,{1,_,_}],{All,2}]][[1]]+4-i,4]];sides[[bottomEdgeNumber[[1]]]][#,2]&@r,
					sides[[bottomEdgeNumber[[1]]]][#,2]&@r
					]
		]
	];
]


(*Trying the algorithm for a horizontally rotated Solved Cube, where the white face is downwards.*)
a3=new[Rubik, k, "Solved Cube v3"]
label[a3]
horizontal[a3]
upperEdges[a3]
cubies[a3]

(*Here, the same, but the white face is in the front, not in the bottom.*)
a4=new[Rubik, k, "Solved Cube v4"]
label[a4]
cubies[a4]
upperEdges[a4]
cubies[a4]


(*The next step is flipping the edge cubies in the top layer to their white face upwards.
In a similar way, the 8 corner cubies can be placed, as well. 
Continuing the middle and bottom layer, the whole cube can be solved, only using these operators and cubie positions.
Below, there is the skeleton of the main logic of the Rubik cube solver:*)

mainLogic[init_List]:=Module[{cube=new[Rubik, init, "Cube"]},
Print[cubies[cube]];
upperEdges[cube];
(*here should continue the solution step-by-step

flippingupperEdges[cube];
placingupperCorners[cube];
middleEdges[cube];
flippingbottomEdges[cube];
swapbottomEdges[cube];
swapbottomCorners[cube];
rotatebottomCorners[cube];

*)
;cubies[cube]]

mainLogic[k]
mainLogic[l]



